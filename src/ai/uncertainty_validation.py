import numpy as np

class RiskAssessor:
    """
    Simplified risk assessor that uses basic heuristics to identify risky actions.
    """
    def __init__(self):
        # Basic risk thresholds
        self.health_threshold = 0.3  # 30% health is risky
        self.ammo_threshold = 0.2    # 20% ammo is risky
        
    def is_high_risk(self, state, action, q_values=None):
        """
        Assess if an action is high-risk
        
        Args:
            state: Current state (numpy array)
            action: Action to assess (int)
            q_values: Optional Q-values for all actions
            
        Returns:
            (is_risky, reason): Tuple of boolean and explanation string
        """
        # Extract agent stats from state (assuming last 3 values are health, ammo, shields)
        agent_health = state[-3]
        agent_ammo = state[-2]
        agent_shields = state[-1]
        
        # Risk factors
        risk_level = 0.0
        risk_reasons = []
        
        # Movement actions (0-3)
        if action <= 3:
            # Movement with very low health
            if agent_health < self.health_threshold / 2:
                risk_level += 0.3
                risk_reasons.append(f"Moving with critical health ({agent_health:.2f})")
        
        # Shooting actions (4-7)
        elif action <= 7:
            # Shooting with low ammo
            if agent_ammo < self.ammo_threshold:
                risk_level += 0.5
                risk_reasons.append(f"Shooting with low ammo ({agent_ammo:.2f})")
                
            # Shooting with low health
            if agent_health < self.health_threshold:
                risk_level += 0.3
                risk_reasons.append(f"Shooting with low health ({agent_health:.2f})")
        
        # Special actions (8-10)
        else:
            # Using special actions when health is critical
            if agent_health < self.health_threshold / 2 and action != 9:  # 9 is "use cover"
                risk_level += 0.3
                risk_reasons.append(f"Using special action with critical health ({agent_health:.2f})")
        
        # Check Q-values if available
        if q_values is not None:
            action_q = q_values[action]
            best_q = np.max(q_values)
            
            # Action is significantly worse than best option
            if best_q - action_q > 0.5 and best_q > 0:
                risk_level += 0.3
                risk_reasons.append(f"Suboptimal action (Q-value: {action_q:.2f} vs best: {best_q:.2f})")
            
            # Action has negative expected value
            if action_q < 0 and best_q > 0:
                risk_level += 0.3
                risk_reasons.append(f"Negative expected value ({action_q:.2f})")
        
        # Determine if risky (threshold of 0.4)
        is_risky = risk_level >= 0.4
        risk_reason = "; ".join(risk_reasons) if risk_reasons else "Low risk action"
        
        return is_risky, risk_reason