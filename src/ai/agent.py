import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F

from src.ai.models import DuelingDQN

class RLAgent:
    """
    Simplified Reinforcement Learning agent using DQN
    """
    def __init__(self, state_size, action_size, replay_buffer_size=10000, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        
        # Initialize model
        self.model = DuelingDQN(state_size, action_size)
        
        # Initialize target model
        self.target_model = DuelingDQN(state_size, action_size)
        self.update_target_model()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Exploration parameters
        self.epsilon = 0.2
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        # Discount factor
        self.gamma = 0.99
        
        # Experience replay
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        
        # Training parameters
        self.target_update_frequency = 10
        self.update_counter = 0
        
        # Action categories for safe action selection
        self.movement_actions = [0, 1, 2, 3]  # up, down, left, right
        self.combat_actions = [4, 5, 6, 7]    # shoot in different directions
        self.special_actions = [8, 9, 10]     # place trap, use cover, call support
        
    def update_target_model(self):
        """Update target model with weights from main model"""
        self.target_model.load_state_dict(self.model.state_dict())
        
    def select_action(self, state, uncertainty_estimator=None, explore=True):
        """Select an action using epsilon-greedy policy"""
        # Exploration
        if explore and random.random() < self.epsilon:
            # Simple balanced exploration strategy
            if random.random() < 0.6:  # Prefer movement
                action = random.choice(self.movement_actions)
            elif random.random() < 0.8:  # Sometimes combat
                action = random.choice(self.combat_actions)
            else:  # Occasionally special actions
                action = random.choice(self.special_actions)
                
            # Get Q-values for logging
            q_values = self.get_q_values(state)
            return action, q_values
        
        # Exploitation - use Q-values
        q_values = self.get_q_values(state)
        return np.argmax(q_values), q_values
    
    def get_q_values(self, state):
        """Get Q-values for a state"""
        state_tensor = torch.FloatTensor(state)
        
        # Add batch dimension if necessary
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
            
        # Check if input size matches model's expected input size
        if state_tensor.shape[1] != self.state_size:
            # Handle size mismatch by resizing the input to match model's expected size
            # This is a temporary fix for the experiment - ideally, the model should match the environment
            print(f"Warning: Input size mismatch. Got {state_tensor.shape[1]}, expected {self.state_size}.")
            # Resize tensor by either padding with zeros or truncating
            if state_tensor.shape[1] < self.state_size:
                # Pad with zeros
                padding = torch.zeros(state_tensor.shape[0], self.state_size - state_tensor.shape[1])
                state_tensor = torch.cat([state_tensor, padding], dim=1)
            else:
                # Truncate
                state_tensor = state_tensor[:, :self.state_size]
            
        with torch.no_grad():
            q_values = self.model(state_tensor)
            
            if isinstance(q_values, torch.Tensor):
                q_values = q_values.cpu().numpy()[0]
                
        return q_values
        
    def select_safe_action(self, state, current_action=None):
        """Select a safe action when veto has been triggered"""
        # Get Q-values for all actions
        q_values = self.get_q_values(state)
        
        # Create action mask to avoid the vetoed action
        mask = np.ones(self.action_size)
        if current_action is not None:
            mask[current_action] = 0
            
        # Prioritize movement actions for safety
        movement_bonus = np.zeros(self.action_size)
        for action in self.movement_actions:
            movement_bonus[action] = 0.5
            
        # Apply mask and bonuses
        adjusted_q_values = q_values * mask + movement_bonus
        
        return np.argmax(adjusted_q_values)
        
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def train(self):
        """Train the model on a batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample random batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Compute Q-values for current states and actions
        current_q_values = self.model(states).gather(1, actions).squeeze(1)
        
        # Compute Q-values for next states using target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            
        # Compute target Q-values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.update_target_model()
            
        return loss.item()
        
    def save(self, path):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load(self, path):
        """Load model from file"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

    def learn_from_veto(self, state, vetoed_action, alternative_action, reward):
        """Simple implementation to learn from veto experiences"""
        # Store vetoed action as a negative example with reduced reward
        self.store_transition(state, vetoed_action, -0.1, state, False)
        
        # Store alternative action as a positive example with the actual reward
        self.store_transition(state, alternative_action, reward, state, False)
        
        # Train on these new examples
        if len(self.replay_buffer) >= self.batch_size:
            self.train()