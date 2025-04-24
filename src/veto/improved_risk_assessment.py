"""
Improved risk assessment module that uses structured GameState objects.
Provides standardized risk evaluation for veto mechanisms.
"""

import numpy as np
from src.game.state import GameState
from src.game.state_adapter import StateAdapter

class ImprovedRiskAssessor:
    """
    Risk assessor that uses structured GameState objects for more reliable assessment.
    This replaces the previous risk assessors with a single source of truth.
    """
    def __init__(self):
        # Risk thresholds
        self.health_threshold = 0.3  # 30% health is risky
        self.critical_health_threshold = 0.15  # 15% health is critical
        self.ammo_threshold = 0.2    # 20% ammo is risky
        
        # Action categories
        self.movement_actions = [0, 1, 2, 3]
        self.combat_actions = [4, 5, 6, 7]
        self.special_actions = [8, 9, 10]
        
        # Action descriptions for better explanations
        self.action_descriptions = {
            0: "Move Up",
            1: "Move Down",
            2: "Move Left",
            3: "Move Right",
            4: "Shoot Up",
            5: "Shoot Down",
            6: "Shoot Left",
            7: "Shoot Right",
            8: "Place Trap",
            9: "Use Cover",
            10: "Call Support"
        }
        
    def is_high_risk(self, state, action, q_values=None):
        """
        Assess if an action is high-risk
        
        Args:
            state: GameState object or raw state array
            action: Action to assess (int)
            q_values: Optional Q-values for all actions
            
        Returns:
            (is_risky, reason): Tuple of boolean and explanation string
        """
        # Convert to GameState if raw array
        if not isinstance(state, GameState):
            state = GameState(raw_state=state)
            
        # Calculate risk factors and reasons
        risk_level = 0.0
        risk_reasons = []
        
        # Check based on action type
        if action in self.movement_actions:
            risk_level, reasons = self._assess_movement_risk(state, action)
            risk_reasons.extend(reasons)
        elif action in self.combat_actions:
            risk_level, reasons = self._assess_combat_risk(state, action)
            risk_reasons.extend(reasons)
        elif action in self.special_actions:
            risk_level, reasons = self._assess_special_action_risk(state, action)
            risk_reasons.extend(reasons)
        
        # Check Q-values if available
        if q_values is not None:
            q_risk, q_reasons = self._assess_q_value_risk(q_values, action)
            risk_level += q_risk
            risk_reasons.extend(q_reasons)
        
        # Determine if risky (threshold of 0.4)
        is_risky = risk_level >= 0.4
        risk_reason = "; ".join(risk_reasons) if risk_reasons else "Low risk action"
        
        return is_risky, risk_reason
    
    def _assess_movement_risk(self, state, action):
        """Assess risk for movement actions"""
        risk_level = 0.0
        reasons = []
        
        # Very low health
        if state.is_low_health(self.critical_health_threshold):
            risk_level += 0.3
            reasons.append(f"Moving with critical health ({state.health:.2f})")
            
        return risk_level, reasons
    
    def _assess_combat_risk(self, state, action):
        """Assess risk for combat actions"""
        risk_level = 0.0
        reasons = []
        
        # Low ammo
        if state.is_low_ammo(self.ammo_threshold):
            risk_level += 0.5
            reasons.append(f"Shooting with low ammo ({state.ammo:.2f})")
            
        # Low health
        if state.is_low_health(self.health_threshold):
            risk_level += 0.3
            reasons.append(f"Shooting with low health ({state.health:.2f})")
            
        # No shields and low health
        if state.is_low_health(self.health_threshold) and not state.has_shields():
            risk_level += 0.2
            reasons.append("No shields available for protection")
            
        return risk_level, reasons
    
    def _assess_special_action_risk(self, state, action):
        """Assess risk for special actions"""
        risk_level = 0.0
        reasons = []
        
        # Critical health (except for "use cover" which is action 9)
        if state.is_low_health(self.critical_health_threshold) and action != 9:
            risk_level += 0.3
            reasons.append(f"Using special action with critical health ({state.health:.2f})")
            
        return risk_level, reasons
    
    def _assess_q_value_risk(self, q_values, action):
        """Assess risk based on Q-values"""
        risk_level = 0.0
        reasons = []
        
        # Get Q-value for this action
        action_q = q_values[action]
        
        # Get best Q-value and action
        best_action = np.argmax(q_values)
        best_q = q_values[best_action]
        
        # Action is significantly worse than best option
        if best_q - action_q > 0.5 and best_q > 0:
            risk_level += 0.3
            reasons.append(f"Suboptimal action (Q={action_q:.2f} vs best={best_q:.2f})")
        
        # Action has negative expected value
        if action_q < 0 and best_q > 0:
            risk_level += 0.3
            reasons.append(f"Negative expected value ({action_q:.2f})")
            
        return risk_level, reasons
    
    def extract_features(self, state, action, q_values=None):
        """
        Extract features for risk assessment
        
        Args:
            state: GameState object or raw state array
            action: Action to assess (int)
            q_values: Optional Q-values for all actions
            
        Returns:
            numpy array of features
        """
        # Convert to GameState if raw array
        if not isinstance(state, GameState):
            state = GameState(raw_state=state)
            
        features = []
        
        # 1. Agent stats
        features.extend([state.health, state.ammo, state.shields])
        
        # 2. Action type (one-hot encoded)
        action_features = np.zeros(11)  # Assuming 11 action types
        if action < len(action_features):
            action_features[action] = 1
        features.extend(action_features)
        
        # 3. Action category features
        is_movement = 1 if action in self.movement_actions else 0
        is_combat = 1 if action in self.combat_actions else 0
        is_special = 1 if action in self.special_actions else 0
        features.extend([is_movement, is_combat, is_special])
        
        # 4. Resource combination features
        low_health = 1 if state.is_low_health(self.health_threshold) else 0
        low_ammo = 1 if state.is_low_ammo(self.ammo_threshold) else 0
        no_shields = 0 if state.has_shields() else 1
        
        features.extend([low_health, low_ammo, no_shields])
        
        # 5. Interaction features
        combat_low_health = is_combat * low_health
        combat_low_ammo = is_combat * low_ammo
        move_critical_health = is_movement * (1 if state.is_low_health(self.critical_health_threshold) else 0)
        
        features.extend([combat_low_health, combat_low_ammo, move_critical_health])
        
        # 6. Q-value features if available
        if q_values is not None:
            # Current action Q-value
            action_q = q_values[action]
            # Best action and its Q-value
            best_action = np.argmax(q_values)
            best_q = q_values[best_action]
            # Difference from best Q-value
            q_diff = best_q - action_q
            # Normalized rank of this action (0 = best, 1 = worst)
            sorted_actions = np.argsort(-q_values)  # Descending sort
            rank = np.where(sorted_actions == action)[0][0]
            normalized_rank = rank / (len(q_values) - 1)
            
            features.extend([action_q, best_q, q_diff, normalized_rank])
            
            # Is this the best action?
            is_best_action = 1 if action == best_action else 0
            features.append(is_best_action)
        else:
            # Placeholders if Q-values not available
            features.extend([0, 0, 0, 0, 0])
        
        return np.array(features)
        
    def get_action_description(self, action):
        """Get human-readable description of action"""
        return self.action_descriptions.get(action, f"Unknown Action {action}")