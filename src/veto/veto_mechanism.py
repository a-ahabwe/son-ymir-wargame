import time
import numpy as np
from src.veto.risk_assessment import RiskAssessor

class VetoDecision:
    """Class to store the results of a veto assessment"""
    def __init__(self, 
                 original_action, 
                 vetoed, 
                 reason="", 
                 risk_reason="", 
                 alternative_action=None, 
                 uncertainty=None, 
                 threshold=None, 
                 q_values=None):
        """
        Initialize a veto decision
        
        Args:
            original_action: The original action that was assessed
            vetoed: Boolean indicating if the action was vetoed
            reason: Primary reason for the veto
            risk_reason: Detailed risk assessment reason
            alternative_action: The alternative action chosen
            uncertainty: Uncertainty estimate for the action if available
            threshold: Threshold value used for decision if applicable
            q_values: Q-values for the state if available
        """
        self.original_action = original_action
        self.vetoed = vetoed
        self.reason = reason
        self.risk_reason = risk_reason
        self.alternative_action = alternative_action
        self.uncertainty = uncertainty
        self.threshold = threshold
        self.q_values = q_values
        self.timestamp = time.time()

class VetoMechanism:
    """Base class for veto mechanisms"""
    def __init__(self, timeout=10):
        self.timeout = timeout
        self.risk_assessor = RiskAssessor()
        self.veto_history = []
        
    def assess_action(self, state, action, q_values=None, uncertainty=None):
        """
        Determine if veto is needed
        Returns VetoDecision object with assessment results
        """
        is_risky, risk_reason = self.risk_assessor.is_high_risk(state, action, q_values)
        
        if is_risky:
            return VetoDecision(
                original_action=action,
                vetoed=True,
                reason="AI strategy assessment",
                risk_reason=risk_reason,
                q_values=q_values,
                uncertainty=uncertainty
            )
        else:
            return VetoDecision(
                original_action=action,
                vetoed=False,
                q_values=q_values,
                uncertainty=uncertainty
            )
            
    def record_veto_decision(self, state, action, vetoed, alternative=None, outcome=None):
        """Record veto decision and outcome for analysis"""
        self.veto_history.append({
            'state': state,
            'action': action,
            'vetoed': vetoed,
            'alternative': alternative,
            'outcome': outcome,
            'timestamp': time.time()
        })
        
class ThresholdVetoMechanism(VetoMechanism):
    """
    Threshold-based veto mechanism.
    Uses simple threshold on risk assessment to trigger veto.
    """
    def __init__(self, threshold=0.7, timeout=10, model_path=None):
        super().__init__(timeout)
        self.threshold = threshold
        self.risk_assessor = RiskAssessor(model_path)
        
    def assess_action(self, state, action, q_values=None, uncertainty=None):
        """Determine if veto is needed using threshold"""
        is_risky, risk_reason = self.risk_assessor.is_high_risk(state, action, q_values)
        
        if is_risky:
            return VetoDecision(
                original_action=action,
                vetoed=True,
                reason="AI identified a potentially risky action",
                risk_reason=risk_reason,
                q_values=q_values,
                uncertainty=uncertainty,
                threshold=self.threshold
            )
        else:
            return VetoDecision(
                original_action=action,
                vetoed=False,
                q_values=q_values,
                uncertainty=uncertainty,
                threshold=self.threshold
            )
            
class UncertaintyVetoMechanism(VetoMechanism):
    """
    Uncertainty-based veto mechanism.
    Triggers veto when model uncertainty is high.
    
    Enhanced with adaptive thresholding for better veto decisions
    """
    def __init__(self, uncertainty_estimator, uncertainty_threshold=0.7, timeout=10, use_adaptive_threshold=True):
        super().__init__(timeout)
        self.uncertainty_estimator = uncertainty_estimator
        self.initial_uncertainty_threshold = uncertainty_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        self.threshold_history = []  # Track threshold values over time
        self.consecutive_veto_count = 0  # Track consecutive vetoes to avoid excessive vetoing
        self.max_consecutive_vetoes = 5  # Increased to allow more consecutive vetoes before throttling
        
        # Veto feedback learning
        self.veto_outcome_history = []  # Track veto outcomes for learning
        self.max_veto_history = 100     # Maximum history to track
        self.action_veto_frequency = np.zeros(11)  # Track veto frequency by action type
        self.action_count = np.ones(11)  # Count of actions (start at 1 to avoid division by zero)
        
    def assess_action(self, state, action, q_values=None, uncertainty=None):
        """Determine if veto is needed based on uncertainty"""
        # Use veto history to adapt thresholds for specific action types
        action_specific_threshold = self._get_action_specific_threshold(action)
        
        # Update threshold if adaptive thresholding is enabled
        if self.use_adaptive_threshold and hasattr(self.uncertainty_estimator, 'adaptive_uncertainty_threshold'):
            self.uncertainty_threshold = self.uncertainty_estimator.adaptive_uncertainty_threshold()
            self.threshold_history.append(self.uncertainty_threshold)
            
            # Fallback to initial threshold if adaptive threshold is too high or too low
            if self.uncertainty_threshold > 0.8 or self.uncertainty_threshold < 0.1:
                self.uncertainty_threshold = self.initial_uncertainty_threshold
        
        # Check base risk assessment
        is_risky, risk_reason = self.risk_assessor.is_high_risk(state, action, q_values)
        
        # Use provided uncertainty estimate or calculate if not provided
        if uncertainty is None and hasattr(self.uncertainty_estimator, 'decision_uncertainty'):
            uncertainty = self.uncertainty_estimator.decision_uncertainty(state, action)
        
        # Handle case where uncertainty might be an array
        if isinstance(uncertainty, np.ndarray):
            if uncertainty.size > 1:
                # If it's an array with multiple values, take the mean 
                uncertainty_value = uncertainty.mean()
            else:
                # If it's a single-element array, extract the value
                uncertainty_value = uncertainty.item()
        else:
            # If it's already a scalar, use it directly
            uncertainty_value = uncertainty
            
        # Apply action-specific threshold adjustment, with a lower factor to increase veto likelihood
        # FIX 1: Add a minimum threshold to prevent it from being too low
        adjusted_threshold = max(0.1, self.uncertainty_threshold * action_specific_threshold * 0.8)  # Added 0.8 multiplier
        
        # Less aggressive adjustment for consecutive veto prevention
        if self.consecutive_veto_count >= self.max_consecutive_vetoes:
            # Temporarily increase threshold to allow some actions through, but less than before
            adjusted_threshold = adjusted_threshold * 1.2  # Reduced from 1.5
        
        # DEBUG: Print uncertainty and threshold values
        print(f"DEBUG: Action {action} - Uncertainty: {uncertainty_value:.4f}, Threshold: {adjusted_threshold:.4f}")
        
        # Create veto decision object
        veto_decision = VetoDecision(
            original_action=action,
            vetoed=False,
            q_values=q_values,
            uncertainty=uncertainty_value,
            threshold=adjusted_threshold
        )
        
        # Determine if we should veto based primarily on uncertainty
        should_veto = False
        
        if uncertainty_value is not None and uncertainty_value > adjusted_threshold:
            # Primary veto reason: AI uncertainty
            should_veto = True
            veto_decision.reason = "AI is uncertain about this action"
            veto_decision.risk_reason = f"High uncertainty in action outcome ({uncertainty_value:.2f})"
            # DEBUG: Print uncertainty veto triggered
            print(f"DEBUG: Uncertainty veto triggered! Uncertainty: {uncertainty_value:.4f} > Threshold: {adjusted_threshold:.4f}")
        elif is_risky and ((action >= 4 and action <= 7 and "health" in risk_reason.lower()) or 
                          (action > 7 and "resource" in risk_reason.lower()) or
                          "suboptimal" in risk_reason.lower()):  # Added suboptimal condition
            # Secondary veto reason: Risk assessment for shooting with low health or
            # special actions with insufficient resources
            should_veto = True
            veto_decision.reason = "Risk assessment suggests avoiding this action"
            veto_decision.risk_reason = risk_reason
            # DEBUG: Print risk-based veto triggered
            print(f"DEBUG: Risk-based veto triggered! Risk reason: {risk_reason}")
        
        # Explicitly set the vetoed property based on should_veto
        veto_decision.vetoed = should_veto
        
        # Track consecutive vetoes to prevent excessive vetoing
        if should_veto:
            self.consecutive_veto_count += 1
            # Update action veto frequency
            self.action_veto_frequency[action] += 1
            # DEBUG: Print veto count
            print(f"DEBUG: Veto applied. Consecutive veto count: {self.consecutive_veto_count}")
        else:
            # Reset consecutive veto count when an action passes
            self.consecutive_veto_count = 0
            
        # Update action count
        self.action_count[action] += 1
            
        return veto_decision
        
    def _get_action_specific_threshold(self, action):
        """
        Calculate action-specific threshold multiplier based on veto history
        
        This adjusts thresholds based on historical veto frequency for each action type
        """
        # Lower base multiplier to increase veto likelihood
        base_multiplier = 0.9  # Reduced from 1.0
        
        # Calculate veto rate for this action type
        veto_rate = self.action_veto_frequency[action] / self.action_count[action]
        
        # If an action is frequently vetoed, gradually increase its threshold
        if veto_rate > 0.6:  # Increased from 0.5 to reduce threshold increases
            # Increase threshold by up to 20% for very high veto rates (reduced from 30%)
            base_multiplier *= 1.0 + (veto_rate - 0.6) * 0.4  # Reduced from 0.6
        elif veto_rate < 0.15:  # Increased from 0.1 to apply reduction more often
            # Decrease threshold by up to 25% for actions rarely vetoed (increased from 20%)
            base_multiplier *= 0.75 + veto_rate * 1.5  # More reduction (changed from 0.8 + rate*2.0)
            
        return base_multiplier
        
    def record_veto_decision(self, state, action, vetoed, alternative=None, outcome=None):
        """
        Enhanced record function that updates veto history and learns from outcomes
        """
        # Call parent method to record basic info
        super().record_veto_decision(state, action, vetoed, alternative, outcome)
        
        # Record outcome info for learning
        if outcome is not None:
            # Add to outcome history
            veto_outcome = {
                'action': action,
                'vetoed': vetoed,
                'alternative': alternative,
                'outcome': outcome,
                'timestamp': time.time()
            }
            
            # Maintain history size
            if len(self.veto_outcome_history) >= self.max_veto_history:
                self.veto_outcome_history.pop(0)
            
            self.veto_outcome_history.append(veto_outcome)
            
            # Use outcome to update thresholds
            self._update_from_outcome(veto_outcome)
            
    def _update_from_outcome(self, veto_outcome):
        """
        Update veto parameters based on outcome
        
        If veto led to better outcome, reinforce current behavior
        If veto led to worse outcome, adjust thresholds
        """
        # Extract info
        action = veto_outcome['action']
        vetoed = veto_outcome['vetoed']
        outcome = veto_outcome['outcome']
        
        # Simple outcome analysis - assuming outcome[0] is reward
        reward = outcome[0] if isinstance(outcome, tuple) else outcome
        
        # Positive reward after veto suggests good veto decision
        if vetoed and reward > 0:
            # Slightly lower threshold to encourage this type of veto
            self.action_veto_frequency[action] += 0.1  # Small reinforcement
            
        # Negative reward after veto suggests bad veto decision
        elif vetoed and reward < 0:
            # Increase threshold to discourage this type of veto
            if self.action_veto_frequency[action] > 0:
                self.action_veto_frequency[action] -= 0.2  # Larger negative reinforcement
                
        # Non-vetoed action with high reward suggests good non-veto decision
        elif not vetoed and reward > 1.0:
            # Increase threshold to allow more of these actions
            self.action_count[action] += 0.5  # Effectively reducing veto rate