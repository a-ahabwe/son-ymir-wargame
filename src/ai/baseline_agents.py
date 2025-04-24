import numpy as np
import random
from collections import deque

class RandomAgent:
    """
    Baseline agent that selects actions randomly.
    Serves as an absolute lower bound for performance.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
    def select_action(self, state, uncertainty_estimator=None):
        """Select a random action"""
        action = random.randint(0, self.action_size - 1)
        q_values = np.zeros(self.action_size)  # Placeholder for Q-values
        return action, q_values
        
    def store_transition(self, state, action, reward, next_state, done):
        """Do nothing - random agent doesn't learn"""
        pass
        
    def train(self):
        """Do nothing - random agent doesn't learn"""
        pass
        
    def save(self, path):
        """Save agent (no-op for random agent)"""
        pass
        
    def load(self, path):
        """Load agent (no-op for random agent)"""
        pass

class RuleBasedAgent:
    """
    Baseline agent using simple rule-based approaches.
    Uses predefined rules rather than learning from experience.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.grid_size = int(np.sqrt(state_size - 3))  # Subtract 3 for agent stats
        
        # Action mapping
        self.movement_actions = [0, 1, 2, 3]  # up, down, left, right
        self.combat_actions = [4, 5, 6, 7]    # shoot in different directions
        self.special_actions = [8, 9, 10]     # place trap, use cover, call support
        
        # State tracking
        self.last_action = None
        self.action_count = {i: 0 for i in range(action_size)}
        
    def select_action(self, state, uncertainty_estimator=None):
        """
        Select action based on simple rules:
        - If health is low, prioritize movement and cover
        - If ammo is low, prioritize movement to find resources
        - Otherwise, prioritize combat actions
        - Avoid repeating the same action too many times
        """
        # Extract agent stats from state
        agent_health = state[-3]
        agent_ammo = state[-2]
        agent_shields = state[-1]
        
        # Generate placeholder Q-values based on rules
        q_values = np.zeros(self.action_size)
        
        # Set base preferences
        for i in self.movement_actions:
            q_values[i] = 0.5  # Base value for movement
        for i in self.combat_actions:
            q_values[i] = 0.3  # Base value for combat
        for i in self.special_actions:
            q_values[i] = 0.2  # Base value for special actions
        
        # Adjust based on state
        if agent_health < 0.3:  # Low health
            # Prioritize movement and cover
            for i in self.movement_actions:
                q_values[i] += 0.4
            q_values[9] += 0.5  # Cover action
            # Reduce combat priority
            for i in self.combat_actions:
                q_values[i] -= 0.2
        elif agent_ammo < 0.2:  # Low ammo
            # Prioritize movement to find resources
            for i in self.movement_actions:
                q_values[i] += 0.3
            # Reduce combat priority
            for i in self.combat_actions:
                q_values[i] -= 0.3
        else:  # Normal state
            # Prioritize combat
            for i in self.combat_actions:
                q_values[i] += 0.3
                
        # Avoid repeating the same action
        if self.last_action is not None:
            repeat_count = self.action_count[self.last_action]
            if repeat_count > 2:
                q_values[self.last_action] -= 0.3 * repeat_count
                
        # Select the best action according to rule-based Q-values
        action = np.argmax(q_values)
        
        # Update action tracking
        self.last_action = action
        self.action_count[action] += 1
        for a in self.action_count:
            if a != action:
                self.action_count[a] = max(0, self.action_count[a] - 0.5)
                
        return action, q_values
        
    def store_transition(self, state, action, reward, next_state, done):
        """Do nothing - rule-based agent doesn't learn from experience"""
        pass
        
    def train(self):
        """Do nothing - rule-based agent doesn't learn"""
        pass
        
    def save(self, path):
        """Save agent (no-op for rule-based agent)"""
        pass
        
    def load(self, path):
        """Load agent (no-op for rule-based agent)"""
        pass

class SimpleQLearningAgent:
    """
    Baseline Q-learning agent with tabular representation.
    Much simpler than DQN, serves as a baseline for learning approaches.
    """
    def __init__(self, state_size, action_size, learning_rate=0.1, gamma=0.99, epsilon=0.2):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        # Simplified state representation
        # Use bucketed agent stats (3 values) for simpler state space
        self.q_table = {}
        
    def _get_simplified_state(self, state):
        """Convert continuous state to discretized version for table lookup"""
        # Extract agent stats
        health = state[-3]
        ammo = state[-2]
        shields = state[-1]
        
        # Discretize health into 3 buckets: low, medium, high
        if health < 0.3:
            health_bucket = 0
        elif health < 0.7:
            health_bucket = 1
        else:
            health_bucket = 2
            
        # Discretize ammo into 3 buckets
        if ammo < 0.2:
            ammo_bucket = 0
        elif ammo < 0.6:
            ammo_bucket = 1
        else:
            ammo_bucket = 2
            
        # Discretize shields into 2 buckets
        shield_bucket = 1 if shields > 0 else 0
        
        # Create a tuple key for the Q-table
        return (health_bucket, ammo_bucket, shield_bucket)
        
    def select_action(self, state, uncertainty_estimator=None):
        """Select action using epsilon-greedy policy"""
        simplified_state = self._get_simplified_state(state)
        
        # Ensure state exists in Q-table
        if simplified_state not in self.q_table:
            self.q_table[simplified_state] = np.zeros(self.action_size)
            
        # Get Q-values for this state
        q_values = self.q_table[simplified_state]
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
        else:
            action = np.argmax(q_values)
            
        return action, q_values
        
    def store_transition(self, state, action, reward, next_state, done):
        """Update Q-table with new experience"""
        # Get simplified states
        state_key = self._get_simplified_state(state)
        next_state_key = self._get_simplified_state(next_state)
        
        # Ensure states exist in Q-table
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
            
        # Get current and next Q-values
        q_values = self.q_table[state_key]
        next_q_values = self.q_table[next_state_key]
        
        # Q-learning update
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(next_q_values)
            
        # Update Q-value for (state, action)
        q_values[action] = q_values[action] + self.learning_rate * (target - q_values[action])
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def train(self):
        """Q-learning training is done in store_transition"""
        pass
        
    def save(self, path):
        """Save Q-table to file"""
        import json
        import os
        
        # Convert Q-table keys to strings for JSON
        serializable_q_table = {}
        for k, v in self.q_table.items():
            serializable_q_table[str(k)] = v.tolist()
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save to file
        with open(path, 'w') as f:
            json.dump({
                'q_table': serializable_q_table,
                'epsilon': self.epsilon
            }, f)
            
    def load(self, path):
        """Load Q-table from file"""
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Convert string keys back to tuples
        self.q_table = {}
        for k, v in data['q_table'].items():
            # Convert string representation of tuple back to actual tuple
            # Format is like "(0, 1, 2)"
            key = tuple(map(int, k.strip('()').split(',')))
            self.q_table[key] = np.array(v)
            
        self.epsilon = data['epsilon']