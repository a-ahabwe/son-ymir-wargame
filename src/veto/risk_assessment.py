import numpy as np
import joblib
import time

class RiskAssessor:
    """Assesses the risk of actions"""
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            try:
                self.model = joblib.load(model_path)
            except:
                print(f"Warning: Could not load risk model from {model_path}")
                
        # Cache for risk assessments
        self.risk_cache = {}
        self.cache_timeout = 5.0  # Cache entries expire after 5 seconds
        
        # Adaptive risk threshold tracking
        self.risk_history = []
        self.max_risk_history = 100
        self.risk_thresholds = {
            "health": 0.3,  # Initial health threshold
            "ammo": 1.0,    # Initial ammo threshold
            "shield": 0.0   # Initial shield threshold
        }
        
        # Risk statistics by action type
        self.action_risk_stats = {}  # Will store {action_type: {"count": N, "total_risk": X}}
        
    def extract_features(self, state, action, q_values=None):
        """
        Extract features for risk assessment
        Features:
        - Agent health
        - Agent ammo
        - Agent shields
        - Action type
        - Q-value of action
        - Q-value difference from best action
        - Q-value difference from average
        """
        features = []
        
        # Agent stats are the last 3 elements of the state
        agent_health = state[-3]
        agent_ammo = state[-2]
        agent_shields = state[-1]
        
        features.append(agent_health)
        features.append(agent_ammo)
        features.append(agent_shields)
        
        # One-hot encode action type
        action_type = np.zeros(11)
        action_type[action] = 1
        features.extend(action_type)
        
        # Q-value features if available
        if q_values is not None:
            action_q = q_values[action]
            best_q = np.max(q_values)
            avg_q = np.mean(q_values)
            
            features.append(action_q)
            features.append(best_q - action_q)  # Difference from best
            features.append(action_q - avg_q)   # Difference from average
        else:
            # Placeholder values if Q-values not available
            features.extend([0, 0, 0])
            
        return np.array(features)
        
    def is_high_risk(self, state, action, q_values=None):
        """Assess if an action is high-risk"""
        # Check cache first
        cache_key = (tuple(state.flatten()), action)
        current_time = time.time()
        
        if cache_key in self.risk_cache:
            entry_time, result, reason = self.risk_cache[cache_key]
            if current_time - entry_time < self.cache_timeout:
                return result, reason
                
        # Extract features
        features = self.extract_features(state, action, q_values)
        
        # Without a trained model, use heuristics
        if self.model is None:
            result, reason = self._heuristic_risk_assessment(state, action, q_values, features)
        else:
            # Use trained model
            prediction = self.model.predict([features])[0]
            result = prediction == 1
            reason = "Model predicts high risk" if result else "Model predicts low risk"
            
        # Cache the result
        self.risk_cache[cache_key] = (current_time, result, reason)
        
        return result, reason
        
    def _heuristic_risk_assessment(self, state, action, q_values, features):
        """Heuristic risk assessment when no model is available"""
        # Get agent stats
        agent_health = features[0]
        agent_ammo = features[1]
        agent_shields = features[2]
        
        # Initial risk assessment
        risk_level = 0.0
        risk_reasons = []
        
        # Update adaptive thresholds based on current state distribution
        self._update_adaptive_thresholds(agent_health, agent_ammo, agent_shields)
        
        # Basic resource-based risk heuristics with adaptive thresholds
        if action >= 4 and action <= 7 and agent_ammo <= self.risk_thresholds["ammo"]:  
            # Shoot actions with low ammo
            ammo_risk = (self.risk_thresholds["ammo"] - agent_ammo) * 0.6
            risk_level += ammo_risk
            risk_reasons.append(f"Low ammo ({agent_ammo:.1f}) for shooting action")
            
        if agent_health <= self.risk_thresholds["health"]:  # Low health
            health_risk = (self.risk_thresholds["health"] - agent_health) * 1.5
            
            if action != 9:  # Not using cover
                # Apply health risk but with context awareness
                if action >= 0 and action <= 3:  # Movement actions
                    risk_level += health_risk * 0.6  # Reduced risk for movement
                else:
                    risk_level += health_risk
                risk_reasons.append(f"Low health ({agent_health:.1f}) without using cover")
                
            if action >= 4 and action <= 7:  # Shooting with low health
                risk_level += health_risk * 0.4
                risk_reasons.append(f"Aggressive action with health at {agent_health:.1f}")
                
        # Context-based risk modifiers based on action type and Q-values
        if q_values is not None:
            # Use Q-value information to assess relative risk
            action_q = q_values[action]
            best_q = np.max(q_values)
            avg_q = np.mean(q_values)
            
            # If the action's Q-value is significantly worse than the best option
            if best_q - action_q > 1.0 and best_q > 0:
                suboptimal_risk = (best_q - action_q) / best_q * 0.2
                risk_level += suboptimal_risk
                risk_reasons.append("Significantly suboptimal action based on Q-values")
                
            # If the action's Q-value is negative and other options are positive
            if action_q < 0 and best_q > 0.5:
                negative_q_risk = 0.3
                risk_level += negative_q_risk
                risk_reasons.append("Negative expected value action")
        
        # Learn from state context for movement actions
        if state is not None and action <= 3:
            try:
                # Be more conservative with movement assessment
                if agent_health < self.risk_thresholds["health"] * 0.5 and agent_ammo < self.risk_thresholds["ammo"] * 0.5:
                    # Very low resources situation
                    risk_level += 0.1
                    risk_reasons.append(f"Moving with critical resource levels (H:{agent_health:.1f}, A:{agent_ammo:.0f})")
                    
                # Check if moving without shields in low health situation
                if agent_shields <= self.risk_thresholds["shield"] and agent_health < self.risk_thresholds["health"]:
                    # Moving without shields can be risky when health is also low
                    shield_health_risk = 0.1 * (self.risk_thresholds["health"] - agent_health) * 2 
                    risk_level += shield_health_risk
                    risk_reasons.append(f"Moving without defensive resources (H:{agent_health:.1f}, S:{agent_shields:.0f})")
            except:
                # Fallback if state parsing fails
                pass
                
        # Track risk statistics for this action type
        if action not in self.action_risk_stats:
            self.action_risk_stats[action] = {"count": 0, "total_risk": 0.0}
        self.action_risk_stats[action]["count"] += 1
        self.action_risk_stats[action]["total_risk"] += risk_level
        
        # Adapt threshold based on historical risk for this action type
        final_threshold = self._get_adaptive_risk_threshold(action)
                
        # Final risk assessment
        is_risky = risk_level >= final_threshold
        risk_reason = "; ".join(risk_reasons) if risk_reasons else "Unknown risk"
        
        return is_risky, risk_reason
        
    def _update_adaptive_thresholds(self, health, ammo, shields):
        """Update adaptive thresholds based on agent state"""
        # Store current state in history
        state_record = {
            "health": health,
            "ammo": ammo,
            "shields": shields,
            "timestamp": time.time()
        }
        
        # Manage history size
        if len(self.risk_history) >= self.max_risk_history:
            self.risk_history.pop(0)
        self.risk_history.append(state_record)
        
        # Not enough data yet
        if len(self.risk_history) < 10:
            return
            
        # Calculate adaptive thresholds based on recent history
        health_values = [entry["health"] for entry in self.risk_history[-20:]]
        ammo_values = [entry["ammo"] for entry in self.risk_history[-20:]]
        shield_values = [entry["shields"] for entry in self.risk_history[-20:]]
        
        # Set thresholds to be approximately 25th percentile of recent values
        if health_values:
            self.risk_thresholds["health"] = max(0.2, np.percentile(health_values, 25))
        if ammo_values:
            self.risk_thresholds["ammo"] = max(1, np.percentile(ammo_values, 25))
        if shield_values:
            self.risk_thresholds["shield"] = max(0, np.percentile(shield_values, 25))
            
    def _get_adaptive_risk_threshold(self, action):
        """Get an adaptive risk threshold based on action type and history"""
        # Start with a lower base threshold to increase veto frequency
        base_threshold = 0.5  # Reduced from 0.7
        
        # If we have history for this action, adjust threshold
        if action in self.action_risk_stats and self.action_risk_stats[action]["count"] > 0:
            action_stats = self.action_risk_stats[action]
            avg_risk = action_stats["total_risk"] / action_stats["count"]
            
            # Adjust threshold based on historical average risk for this action
            # Actions with historically high risk get a higher threshold to avoid excessive vetoing
            if avg_risk > 0.4:  # Lowered from 0.5
                # Scale up threshold for high-risk actions, but less than before
                adjusted_threshold = base_threshold * (1.0 + (avg_risk - 0.4) * 0.5)  # Reduced multiplier
                return min(0.8, adjusted_threshold)  # Cap lowered to 0.8
            elif avg_risk < 0.3:
                # Scale down threshold for low-risk actions
                adjusted_threshold = base_threshold * (0.9 - (0.3 - avg_risk) * 0.5)
                return max(0.4, adjusted_threshold)  # Floor lowered to 0.4
                
        return base_threshold