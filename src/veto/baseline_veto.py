import random
import numpy as np
import time
from src.veto.veto_mechanism import VetoDecision, VetoMechanism

class RandomVetoMechanism(VetoMechanism):
    """
    Baseline veto mechanism that randomly vetoes actions.
    Serves as a lower bound for veto performance.
    """
    def __init__(self, veto_probability=0.2, timeout=10):
        super().__init__(timeout)
        self.veto_probability = veto_probability
        
    def assess_action(self, state, action, q_values=None):
        """Randomly veto actions based on probability"""
        should_veto = random.random() < self.veto_probability
        
        return VetoDecision(
            original_action=action,
            vetoed=should_veto,
            reason="Random veto - baseline mechanism" if should_veto else ""
        )

class FixedRulesVetoMechanism(VetoMechanism):
    """
    Baseline veto mechanism using simple fixed rules.
    Uses a combination of basic heuristics without learning or uncertainty.
    """
    def __init__(self, timeout=10):
        super().__init__(timeout)
        
    def assess_action(self, state, action, q_values=None):
        """
        Apply fixed rules for vetoing:
        - Veto shooting actions when ammo is low
        - Veto special actions when health is low
        - Veto combat actions when health is critical
        """
        # Extract agent stats from state
        agent_health = state[-3]
        agent_ammo = state[-2]
        agent_shields = state[-1]
        
        # Default - don't veto
        should_veto = False
        reason = ""
        
        # Rule 1: Don't shoot when ammo is very low
        if action >= 4 and action <= 7 and agent_ammo < 0.1:  # Shooting actions with very low ammo
            should_veto = True
            reason = "Low ammo - shooting not recommended"
            
        # Rule 2: Don't use special actions with very low health
        if action >= 8 and action <= 10 and agent_health < 0.2:
            should_veto = True
            reason = "Low health - special action not recommended"
            
        # Rule 3: Prefer defensive actions when health is critical
        if agent_health < 0.1 and action != 9:  # 9 is "take cover"
            should_veto = True
            reason = "Critical health - defensive action recommended"
            
        # Rule 4: Don't engage in combat without shields if health is low
        if agent_health < 0.3 and agent_shields < 0.1 and action >= 4 and action <= 7:
            should_veto = True
            reason = "Low health and no shields - combat not recommended"
        
        return VetoDecision(
            original_action=action,
            vetoed=should_veto,
            reason=reason
        )

class QValueThresholdVeto(VetoMechanism):
    """
    Baseline veto mechanism using only Q-values.
    Vetoes actions if their Q-value is below a threshold compared to the best action.
    """
    def __init__(self, threshold_factor=0.6, timeout=10):
        super().__init__(timeout)
        self.threshold_factor = threshold_factor  # Fraction of best Q-value that's acceptable
        
    def assess_action(self, state, action, q_values=None):
        """Veto actions with Q-values significantly below the best option"""
        if q_values is None:
            # Can't make a decision without Q-values
            return VetoDecision(
                original_action=action,
                vetoed=False,
                reason="No Q-values available"
            )
        
        # Find the best action and its Q-value
        best_action = np.argmax(q_values)
        best_q_value = q_values[best_action]
        current_q_value = q_values[action]
        
        # Only veto if there's a significant difference and the best action is positive
        should_veto = False
        reason = ""
        
        if best_q_value > 0 and current_q_value < best_q_value * self.threshold_factor:
            should_veto = True
            reason = (f"Q-value ({current_q_value:.2f}) significantly lower than "
                     f"best action ({best_q_value:.2f})")
        
        return VetoDecision(
            original_action=action,
            vetoed=should_veto,
            reason=reason
        )

class HistoricalPerformanceVeto(VetoMechanism):
    """
    Baseline veto mechanism based on historical performance of actions.
    Keeps track of rewards for each action and vetoes those with poor history.
    """
    def __init__(self, history_size=50, poor_threshold=0.0, timeout=10):
        super().__init__(timeout)
        self.history_size = history_size
        self.poor_threshold = poor_threshold
        
        # Initialize history tracking
        self.action_history = {}  # Maps action -> list of (reward, timestamp) tuples
        
    def assess_action(self, state, action, q_values=None):
        """Veto actions with poor historical performance"""
        # Initialize history for this action if needed
        if action not in self.action_history:
            self.action_history[action] = []
            
        # Compute the recent average reward for this action
        recent_history = self.action_history[action]
        
        # Remove old entries (older than 5 minutes)
        current_time = time.time()
        recent_history = [(r, t) for r, t in recent_history 
                         if current_time - t < 300]  # 5 minutes
        self.action_history[action] = recent_history
        
        # If we have enough history, check performance
        if len(recent_history) >= 5:  # Need at least 5 data points
            avg_reward = sum(r for r, _ in recent_history) / len(recent_history)
            
            if avg_reward < self.poor_threshold:
                return VetoDecision(
                    original_action=action,
                    vetoed=True,
                    reason=f"Action has poor historical performance (avg reward: {avg_reward:.2f})"
                )
                
        # Default - don't veto
        return VetoDecision(
            original_action=action,
            vetoed=False,
            reason=""
        )
        
    def record_veto_decision(self, state, action, vetoed, alternative=None, outcome=None):
        """Record the outcome to update action history"""
        super().record_veto_decision(state, action, vetoed, alternative, outcome)
        
        # Extract reward from outcome if available
        if outcome is not None:
            reward = outcome[0] if isinstance(outcome, tuple) else outcome
            
            # Record in action history
            if action not in self.action_history:
                self.action_history[action] = []
                
            # Add to history and maintain max size
            self.action_history[action].append((reward, time.time()))
            if len(self.action_history[action]) > self.history_size:
                self.action_history[action].pop(0)
                
            # If an alternative was used, record its outcome too
            if vetoed and alternative is not None:
                if alternative not in self.action_history:
                    self.action_history[alternative] = []
                    
                self.action_history[alternative].append((reward, time.time()))
                if len(self.action_history[alternative]) > self.history_size:
                    self.action_history[alternative].pop(0)