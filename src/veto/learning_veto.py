"""
Learning-based veto mechanism that improves with experience.
Uses a learned risk model instead of hard-coded rules.
"""

import numpy as np
import time
import os
from src.veto.veto_mechanism import VetoDecision, VetoMechanism
from src.veto.learning_risk_assessor import LearningRiskAssessor

class LearningVetoMechanism(VetoMechanism):
    """
    Veto mechanism that learns from experience.
    Combines learned risk assessment with uncertainty estimation.
    """
    def __init__(self, uncertainty_estimator=None, risk_model_path=None, 
                threshold=0.6, timeout=10, enable_learning=True,
                model_type="random_forest"):
        super().__init__(timeout)
        
        # Components
        self.uncertainty_estimator = uncertainty_estimator
        self.risk_assessor = LearningRiskAssessor(
            model_path=risk_model_path,
            model_type=model_type
        )
        
        # Configuration
        self.risk_threshold = threshold
        self.uncertainty_threshold = 0.5
        self.enable_learning = enable_learning
        
        # Adaptive parameters
        self.consecutive_veto_count = 0
        self.max_consecutive_vetoes = 3
        
        # Learning metrics
        self.metrics = {
            "veto_count": 0,
            "successful_veto_count": 0,
            "false_positive_count": 0,  # Vetoed but outcome was positive
            "false_negative_count": 0,  # Not vetoed but outcome was negative
            "successful_veto_rate": 0.0
        }
        
        # Most recent veto decisions for learning
        self.recent_decisions = []
        
    def assess_action(self, state, action, q_values=None, uncertainty=None):
        """
        Determine if veto is needed by combining learned risk assessment with uncertainty
        
        Args:
            state: Current state
            action: Action to assess
            q_values: Optional Q-values for all actions
            uncertainty: Optional pre-computed uncertainty value
            
        Returns:
            VetoDecision object with assessment results
        """
        # Step 1: Get risk assessment from learned model
        is_risky, risk_reason = self.risk_assessor.is_high_risk(state, action, q_values)
        
        # Step 2: Get uncertainty estimate if available
        if uncertainty is None and self.uncertainty_estimator is not None:
            try:
                # Use uncertainty estimator
                uncertainty = self.uncertainty_estimator.get_action_uncertainty(state, action)
            except:
                # Default value if estimator fails
                uncertainty = 0.0
        else:
            # Default if no estimator is available
            uncertainty = 0.0
            
        # Step 3: Determine veto decision based on risk and uncertainty
        should_veto = False
        veto_reason = ""
        
        # Primary decision based on risk assessment
        if is_risky:
            should_veto = True
            veto_reason = f"Risk assessment: {risk_reason}"
            
        # Secondary decision based on uncertainty (if risk assessment didn't trigger)
        if not should_veto and uncertainty > self.uncertainty_threshold:
            should_veto = True
            veto_reason = f"High uncertainty ({uncertainty:.2f} > {self.uncertainty_threshold:.2f})"
            
        # Step 4: Prevent excessive consecutive vetoes
        if should_veto:
            self.consecutive_veto_count += 1
            if self.consecutive_veto_count > self.max_consecutive_vetoes:
                # Allow action through to prevent excessive vetoing
                should_veto = False
                veto_reason = "Veto suppressed to prevent excessive interventions"
        else:
            # Reset consecutive veto count
            self.consecutive_veto_count = 0
            
        # Store decision for metrics
        self.recent_decisions.append({
            'state': state,
            'action': action,
            'q_values': q_values,
            'is_risky': is_risky,
            'uncertainty': uncertainty,
            'vetoed': should_veto,
            'timestamp': time.time()
        })
        
        # Keep only recent decisions
        if len(self.recent_decisions) > 100:
            self.recent_decisions = self.recent_decisions[-100:]
            
        # Update metrics
        if should_veto:
            self.metrics["veto_count"] += 1
            
        # Create veto decision
        return VetoDecision(
            original_action=action,
            vetoed=should_veto,
            reason=veto_reason,
            risk_reason=risk_reason,
            uncertainty=uncertainty,
            threshold=self.risk_threshold,
            q_values=q_values
        )
        
    def record_veto_decision(self, state, action, vetoed, alternative=None, outcome=None):
        """
        Record veto decision and outcome for learning
        
        Args:
            state: State where action was taken
            action: Action that was vetoed/approved
            vetoed: Whether the action was vetoed
            alternative: The alternative action taken if vetoed
            outcome: The reward received after taking action/alternative
        """
        # Call parent method
        super().record_veto_decision(state, action, vetoed, alternative, outcome)
        
        # Skip if learning is disabled or outcome is unknown
        if not self.enable_learning or outcome is None:
            return
            
        # Find this decision in recent decisions
        matching_decisions = [d for d in self.recent_decisions 
                            if d['action'] == action and np.all(d['state'] == state)]
        
        # If not found, create a basic entry
        if not matching_decisions:
            decision_data = {
                'state': state,
                'action': action,
                'q_values': None,
                'is_risky': False,
                'uncertainty': 0.0,
                'vetoed': vetoed,
                'timestamp': time.time()
            }
        else:
            # Use the most recent matching decision
            decision_data = matching_decisions[-1]
            
        # Extract reward from outcome
        reward = outcome[0] if isinstance(outcome, tuple) else outcome
        
        # Determine if this was a good veto decision
        positive_outcome = reward > 0
        negative_outcome = reward < 0
        
        if vetoed and negative_outcome:
            # Successful veto (would have had negative outcome)
            self.metrics["successful_veto_count"] += 1
        elif vetoed and positive_outcome:
            # False positive (vetoed but would have been positive)
            self.metrics["false_positive_count"] += 1
        elif not vetoed and negative_outcome:
            # False negative (didn't veto but had negative outcome)
            self.metrics["false_negative_count"] += 1
            
        # Update successful veto rate
        if self.metrics["veto_count"] > 0:
            self.metrics["successful_veto_rate"] = (
                self.metrics["successful_veto_count"] / self.metrics["veto_count"]
            )
            
        # Feed data to risk assessor for learning
        if self.enable_learning:
            # Record observed outcome for learning
            self.risk_assessor.record_outcome(
                state, 
                action, 
                decision_data.get('q_values'), 
                reward, 
                vetoed
            )
            
    def save_model(self, path):
        """Save learned model to file"""
        model_path = f"{path}_risk_model.joblib"
        self.risk_assessor.save_model(model_path)
        
        # Save metrics separately
        import json
        metrics_path = f"{path}_metrics.json"
        
        with open(metrics_path, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'risk_threshold': self.risk_threshold,
                'uncertainty_threshold': self.uncertainty_threshold,
                'max_consecutive_vetoes': self.max_consecutive_vetoes
            }, f, indent=2)
            
        return model_path
        
    def load_model(self, path):
        """Load learned model from file"""
        model_path = f"{path}_risk_model.joblib"
        metrics_path = f"{path}_metrics.json"
        
        # Load risk assessor model
        self.risk_assessor.load_model(model_path)
        
        # Load metrics if available
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                self.metrics = data.get('metrics', self.metrics)
                self.risk_threshold = data.get('risk_threshold', self.risk_threshold)
                self.uncertainty_threshold = data.get('uncertainty_threshold', self.uncertainty_threshold)
                self.max_consecutive_vetoes = data.get('max_consecutive_vetoes', self.max_consecutive_vetoes)
                
        return model_path
        
    def get_metrics(self):
        """Get current performance metrics"""
        return self.metrics
        
    def get_feature_importances(self):
        """Get feature importances from risk model"""
        return self.risk_assessor.get_feature_importances()
        
    def train_model(self):
        """Explicitly train the risk model"""
        return self.risk_assessor.train_model()
        
    def tune_thresholds(self, target_veto_rate=0.2):
        """
        Tune risk and uncertainty thresholds based on observed data
        
        Args:
            target_veto_rate: Desired rate of vetoes (0.0 to 1.0)
        """
        if not self.recent_decisions:
            return False
            
        # Calculate current veto rate
        current_veto_rate = sum(1 for d in self.recent_decisions if d['vetoed']) / len(self.recent_decisions)
        
        # Adjust thresholds if needed
        if abs(current_veto_rate - target_veto_rate) > 0.05:
            if current_veto_rate > target_veto_rate:
                # Too many vetoes, increase thresholds
                self.risk_threshold = min(0.9, self.risk_threshold + 0.05)
                self.uncertainty_threshold = min(0.9, self.uncertainty_threshold + 0.05)
            else:
                # Too few vetoes, decrease thresholds
                self.risk_threshold = max(0.1, self.risk_threshold - 0.05)
                self.uncertainty_threshold = max(0.1, self.uncertainty_threshold - 0.05)
                
            print(f"Tuned thresholds: Risk={self.risk_threshold:.2f}, Uncertainty={self.uncertainty_threshold:.2f}")
            return True
        
        return False
        
    def get_learning_progress(self):
        """Get a summary of learning progress"""
        return {
            'model_metrics': self.risk_assessor.metrics,
            'veto_metrics': self.metrics,
            'thresholds': {
                'risk': self.risk_threshold,
                'uncertainty': self.uncertainty_threshold
            },
            'samples_collected': len(self.risk_assessor.experience_buffer) if hasattr(self.risk_assessor, 'experience_buffer') else 0
        }