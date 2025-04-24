import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import heapq  # For prioritized replay
import torch.nn.functional as F

from src.ai.models import DuelingDQN

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer
    Stores transitions with priority based on TD error
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment  # Beta annealing
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done, error=None):
        """Store transition with priority"""
        max_priority = self.max_priority if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            
        # Set priority to max_priority for new transitions
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample a batch of transitions with importance sampling weights"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        # Compute sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Increment beta for annealing
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights
        
    def update_priorities(self, indices, errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + 1e-5  # Add small constant to avoid zero priority
            self.max_priority = max(self.max_priority, self.priorities[idx])
            
    def __len__(self):
        return len(self.buffer)

class RLAgent:
    """
    Reinforcement Learning agent
    
    Enhanced with:
    1. Support for distributional RL (C51 algorithm)
    2. Uncertainty-aware decision making
    3. Improved exploration strategies
    4. Prioritized experience replay
    """
    def __init__(self, state_size, action_size, replay_buffer_size=10000, batch_size=32, 
                 distributional=False, uncertainty_driven=False, num_atoms=51, v_min=-10, v_max=10,
                 use_prioritized_replay=True):
        self.state_size = state_size
        self.action_size = action_size
        self.distributional = distributional
        self.uncertainty_driven = uncertainty_driven
        self.use_prioritized_replay = use_prioritized_replay
        self.training_mode = True  # Whether we're in training mode
        
        # Distributional RL parameters
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        if distributional:
            self.support = torch.linspace(v_min, v_max, num_atoms)
            self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Initialize model
        self.model = DuelingDQN(state_size, action_size, distributional, num_atoms, v_min, v_max)
        
        # Initialize target model
        self.target_model = DuelingDQN(state_size, action_size, distributional, num_atoms, v_min, v_max)
        self.update_target_model()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Loss function depends on algorithm type
        if distributional:
            # For distributional RL, we use KL divergence
            self.criterion = self._categorical_loss
        else:
            # For standard DQN, we use MSE
            self.criterion = nn.MSELoss(reduction='none')  # 'none' to support prioritized replay
        
        # Exploration parameters
        self.epsilon = 0.2  # Reduced from 0.3 for more exploitation
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        # Discount factor
        self.gamma = 0.99
        
        # Experience replay
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(replay_buffer_size)
        else:
            self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        
        # Uncertainty tracking
        self.uncertainty_buffer = deque(maxlen=100)
        
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
        
    def select_action(self, state, uncertainty=None, explore=True):
        """
        Select an action using epsilon-greedy policy
        
        If uncertainty_driven is True, uses uncertainty to guide exploration
        """
        # Track recently vetoed actions if available
        vetoed_actions = getattr(self, 'vetoed_actions', [])
        
        # Mix of exploration and exploitation
        if explore:
            # Use a mix of strategies for exploration
            strategy = random.random()
            
            if strategy < 0.3:  # 30% - Random balanced exploration
                # Use action category weighting to ensure balanced exploration
                action_categories = [
                    self.movement_actions,   # Movement actions (0-3)
                    self.combat_actions,     # Combat actions (4-7)
                    self.special_actions     # Special actions (8-10)
                ]
                
                # Select a category with preference for movement
                cat_weights = [0.6, 0.3, 0.1]  # 60% movement, 30% combat, 10% special
                category = random.choices(action_categories, weights=cat_weights)[0]
                
                # Filter out recently vetoed actions if possible
                valid_actions = [a for a in category if a not in vetoed_actions]
                
                # If all actions in category were vetoed, fall back to original category
                if not valid_actions:
                    valid_actions = category
                    
                # Select action from category
                action = random.choice(valid_actions)
                
                # Get Q-values for logging
                with torch.no_grad():
                    if self.distributional:
                        q_values = self.model.get_q_values(state)
                    else:
                        q_values = self.model(state)
                    
                    if isinstance(q_values, torch.Tensor):
                        q_values = q_values.cpu().numpy()[0]
                
                return action, q_values
                
            elif strategy < 0.5:  # 20% - Uncertainty-driven exploration
                if self.uncertainty_driven and uncertainty is not None:
                    # Use uncertainty to bias exploration towards uncertain areas
                    uncertainty_values = uncertainty.get_uncertainty_map(state)
                    
                    # Convert to probabilities using softmax
                    if isinstance(uncertainty_values, np.ndarray):
                        # Add small constant to avoid division by zero
                        probs = uncertainty_values + 1e-5
                        probs = probs / probs.sum()
                        
                        # Reduce probability of recently vetoed actions
                        for vetoed_action in vetoed_actions:
                            if vetoed_action < len(probs):
                                probs[vetoed_action] *= 0.5  # Reduce probability by 50%
                                
                        # Re-normalize
                        if probs.sum() > 0:
                            probs = probs / probs.sum()
                        
                        # Sample action according to uncertainty probabilities
                        action = np.random.choice(self.action_size, p=probs)
                    else:
                        # Fallback to uniform random
                        action = random.randrange(self.action_size)
                else:
                    # Uniform random exploration
                    action = random.randrange(self.action_size)
                    
                # Get Q-values for logging
                with torch.no_grad():
                    if self.distributional:
                        q_values = self.model.get_q_values(state)
                    else:
                        q_values = self.model(state)
                    
                    if isinstance(q_values, torch.Tensor):
                        q_values = q_values.cpu().numpy()[0]
                        
                return action, q_values
            
            # Fall through to greedy action selection for remaining 50%
        
        # Greedy action - using Q-values to select the best action
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            if self.distributional:
                q_values = self.model.get_q_values(state_tensor)
            else:
                q_values = self.model(state_tensor)
                
            if isinstance(q_values, torch.Tensor):
                q_values = q_values.cpu().numpy()[0]
        
        # Create a mask for recently vetoed actions to reduce their Q-values
        if vetoed_actions:
            action_mask = np.ones(self.action_size)
            for v_action in vetoed_actions:
                if v_action < self.action_size:
                    action_mask[v_action] = 0.9  # Slightly reduce rather than eliminate
            
            # Apply the mask
            q_values_for_selection = q_values * action_mask
        else:
            q_values_for_selection = q_values.copy()
        
        # Action selection using masked Q-values, but return original Q-values
        return np.argmax(q_values_for_selection), q_values
            
    def select_safe_action(self, state, current_action=None, uncertainty_estimator=None):
        """
        Select a safe action when veto has been triggered
        
        Prioritizes:
        1. Movement actions with low uncertainty
        2. Special actions like 'use cover'
        3. Fallback to the best Q-value action that's not the vetoed one
        """
        # Get Q-values for all actions
        q_values = self.get_q_values(state)
        
        # Get uncertainty for all actions if estimator is provided
        uncertainty_map = None
        if uncertainty_estimator is not None:
            uncertainty_map = uncertainty_estimator.get_uncertainty_map(state)
            
        # Track this action as vetoed to learn from it
        if not hasattr(self, 'vetoed_actions'):
            self.vetoed_actions = []
            
        # Add to recently vetoed actions and maintain a limited history
        if current_action is not None and current_action not in self.vetoed_actions:
            self.vetoed_actions.append(current_action)
            if len(self.vetoed_actions) > 5:
                self.vetoed_actions.pop(0)
        
        # Define safe action selection strategy with weighted probabilities
        if random.random() < 0.7:  # 70% chance to use sophisticated approach
            # Calculate action scores using multi-factor approach
            action_scores = np.zeros(self.action_size)
            
            for action in range(self.action_size):
                # Skip the vetoed action
                if action == current_action:
                    continue
                    
                # Start with Q-value as base score
                action_scores[action] = q_values[action]
                
                # Apply context-specific adjustments
                if action in self.movement_actions:
                    # Prefer movement with a bonus
                    action_scores[action] += 0.5
                    
                    # If uncertainty available, prefer lower uncertainty actions
                    if uncertainty_map is not None:
                        uncertainty = uncertainty_map[action]
                        if uncertainty < 0.5:  # Low uncertainty
                            action_scores[action] += (0.5 - uncertainty) * 2
                        else:  # High uncertainty 
                            action_scores[action] -= (uncertainty - 0.5) * 2
                
                # Apply special action bonuses
                if action == 9:  # Use cover
                    # High bonus for 'use cover' when movement is vetoed
                    if current_action in self.movement_actions:
                        action_scores[action] += 1.0
                elif action == 8:  # Reload
                    # Bonus for reload if Q-value is positive
                    if q_values[action] > 0:
                        action_scores[action] += 0.3
                        
                # Penalize actions similar to the vetoed one
                if current_action in self.combat_actions and action in self.combat_actions:
                    action_scores[action] -= 0.5
                    
            # Get best action based on adjusted scores
            max_score = np.max(action_scores)
            
            # If the max score is very negative, default to movement
            if max_score < -1.0:
                # Choose random movement action
                return random.choice(self.movement_actions)
                
            return np.argmax(action_scores)
        else:
            # 30% chance to use simple prioritized approach
            # Prioritize movement actions
            safe_candidates = self.movement_actions.copy()
            random.shuffle(safe_candidates)
            
            # Remove any movement actions that were recently vetoed
            safe_candidates = [a for a in safe_candidates if a not in self.vetoed_actions]
            
            # If no movement actions are available, consider special actions
            if not safe_candidates:
                safe_candidates = [9, 8, 10]  # Use cover, reload, wait
                # Filter out recently vetoed special actions
                safe_candidates = [a for a in safe_candidates if a not in self.vetoed_actions]
                
            # If still no candidates, use any action except the vetoed one
            if not safe_candidates:
                safe_candidates = [a for a in range(self.action_size) if a != current_action]
                
            # If absolutely no options (should never happen), return a random action
            if not safe_candidates:
                return random.randrange(self.action_size)
                
            # For the final selection, consider:
            # 1. Uncertainty (if available)
            # 2. Q-values
            if uncertainty_map is not None:
                # Get uncertainties for candidates
                candidate_uncertainties = [uncertainty_map[a] for a in safe_candidates]
                
                # Find the lowest uncertainty action
                min_uncertainty_idx = np.argmin(candidate_uncertainties)
                min_uncertainty_action = safe_candidates[min_uncertainty_idx]
                min_uncertainty = candidate_uncertainties[min_uncertainty_idx]
                
                # Get Q-values for candidates
                candidate_q_values = [q_values[a] for a in safe_candidates]
                
                # Find the highest Q-value action
                max_q_idx = np.argmax(candidate_q_values)
                max_q_action = safe_candidates[max_q_idx]
                
                # If the uncertainty difference is significant, prefer low uncertainty
                if min_uncertainty < 0.3:
                    return min_uncertainty_action
                # Otherwise prefer higher Q-value
                else:
                    return max_q_action
            else:
                # Without uncertainty, pick the highest Q-value among candidates
                candidate_q_values = [q_values[a] for a in safe_candidates]
                best_idx = np.argmax(candidate_q_values)
                return safe_candidates[best_idx]
    
    def get_q_values(self, state):
        """Get Q-values for a state"""
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            if self.distributional:
                q_values = self.model.get_q_values(state_tensor)
            else:
                q_values = self.model(state_tensor)
                
            if isinstance(q_values, torch.Tensor):
                q_values = q_values.cpu().numpy()[0]
                
        return q_values
        
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        if self.use_prioritized_replay:
            self.replay_buffer.push(state, action, reward, next_state, done)
        else:
            self.replay_buffer.append((state, action, reward, next_state, done))
        
    def _categorical_loss(self, projected_distribution, target_distribution):
        """
        Calculate categorical loss for distributional RL (C51)
        This is effectively a cross-entropy loss between distributions
        """
        # Cross-entropy loss with target distribution
        loss = -(target_distribution * torch.log(projected_distribution + 1e-8)).sum(-1).mean()
        return loss
        
    def _project_distribution(self, next_q_dist, rewards, dones):
        """
        Project the target distribution for distributional RL
        This handles the distributional Bellman update
        """
        batch_size = rewards.size(0)
        
        # Expand dims for broadcasting
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)
        
        # Compute the projected distribution
        # z_j = r + gamma * z_i (unless done)
        support = self.support.unsqueeze(0).expand(batch_size, self.num_atoms)
        target_z = rewards + (1 - dones) * self.gamma * support
        
        # Clamp to support range
        target_z = target_z.clamp(self.v_min, self.v_max)
        
        # Compute projection
        # Map values to corresponding bucket indices
        b = (target_z - self.v_min) / self.delta_z
        lower_bound = b.floor().long()
        upper_bound = b.ceil().long()
        
        # Handle case where target_z is exactly on a grid point
        indistinguishable = (upper_bound > lower_bound).float()
        lower_bound = lower_bound.clamp(0, self.num_atoms - 1)
        upper_bound = upper_bound.clamp(0, self.num_atoms - 1)
        
        # Distribute probability mass
        projected_dist = torch.zeros_like(next_q_dist)
        
        # Lower bucket contribution
        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).unsqueeze(1).expand(batch_size, self.num_atoms).long()
        projected_dist.view(-1).index_add_(0, (lower_bound + offset).view(-1), 
                                           (next_q_dist * (upper_bound.float() - b) * indistinguishable).view(-1))
        
        # Upper bucket contribution
        projected_dist.view(-1).index_add_(0, (upper_bound + offset).view(-1), 
                                           (next_q_dist * (b - lower_bound.float()) * indistinguishable).view(-1))
        
        return projected_dist
        
    def train(self):
        """Train the model on a batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample batch based on prioritization or random selection
        if self.use_prioritized_replay:
            batch, indices, weights = self.replay_buffer.sample(self.batch_size)
            weights = torch.FloatTensor(weights)
        else:
            # Sample random batch
            batch = random.sample(self.replay_buffer, self.batch_size)
            indices = None
            weights = None
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        if self.distributional:
            # ---- Distributional RL (C51) Training ----
            
            # Get current and next distributions
            _, current_q_dist = self.model(states)
            
            with torch.no_grad():
                # Get best actions from current model (double DQN)
                next_q_values = self.model.get_q_values(next_states)
                next_actions = next_q_values.argmax(dim=1, keepdim=True)
                
                # Get distribution for best actions from target model
                _, next_q_dist = self.target_model(next_states)
                next_dist = next_q_dist.gather(1, next_actions.unsqueeze(-1).expand(-1, -1, self.num_atoms)).squeeze(1)
                
                # Project next distribution
                target_dist = self._project_distribution(next_dist, rewards, dones)
            
            # Extract current distribution for chosen actions
            current_dist = current_q_dist.gather(1, actions.unsqueeze(-1).expand(-1, -1, self.num_atoms)).squeeze(1)
            
            # Calculate element-wise loss for priority updates
            element_wise_loss = -torch.sum(target_dist * torch.log(current_dist + 1e-8), dim=1)
            
            # Apply importance sampling weights if using prioritized replay
            if self.use_prioritized_replay:
                # Apply weights to loss
                loss = (element_wise_loss * weights).mean()
                # Update priorities
                priorities = element_wise_loss.detach().cpu().numpy()
                self.replay_buffer.update_priorities(indices, priorities)
            else:
                loss = element_wise_loss.mean()
        else:
            # ---- Standard DQN Training ----
            
            # Compute Q-values for current states and actions
            current_q_values = self.model(states).gather(1, actions).squeeze(1)
            
            # Compute Q-values for next states using target network
            with torch.no_grad():
                next_q_values = self.target_model(next_states).max(1)[0]
                
            # Compute target Q-values
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
            
            # Compute element-wise loss for priority updates
            element_wise_loss = (current_q_values - target_q_values).pow(2)
            
            # Apply importance sampling weights if using prioritized replay
            if self.use_prioritized_replay:
                # Apply weights to loss
                loss = (element_wise_loss * weights).mean()
                # Update priorities
                priorities = element_wise_loss.detach().cpu().numpy()
                self.replay_buffer.update_priorities(indices, priorities)
            else:
                loss = element_wise_loss.mean()
        
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
            'epsilon': self.epsilon,
            'distributional': self.distributional,
            'v_min': self.v_min,
            'v_max': self.v_max,
            'num_atoms': self.num_atoms
        }, path)
        
    def load(self, path):
        """Load model from file"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        
        # Load distributional parameters if available
        if 'distributional' in checkpoint:
            self.distributional = checkpoint['distributional']
        if 'v_min' in checkpoint:
            self.v_min = checkpoint['v_min']
        if 'v_max' in checkpoint:
            self.v_max = checkpoint['v_max']
        if 'num_atoms' in checkpoint:
            self.num_atoms = checkpoint['num_atoms']

    def end_episode(self, final_state, final_reward):
        """Handle the end of an episode, applying final updates as needed"""
        if self.training_mode:
            # If in training mode, ensure we update based on terminal state
            self.store_terminal_state_info(final_state, final_reward)
            
            # Perform a final training step
            if len(self.replay_buffer) >= self.batch_size:
                self.train()
                
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.update_target_model()
            
    def store_terminal_state_info(self, final_state, final_reward):
        """Store the terminal state information in replay buffer"""
        # Check if we have previous state information to form a complete transition
        if hasattr(self, 'last_state') and hasattr(self, 'last_action'):
            # Store transition with terminal flag
            self.store_transition(
                self.last_state,
                self.last_action,
                final_reward,
                final_state,
                True  # Terminal state
            )

    def learn_from_veto(self, state, vetoed_action, alternative_action, reward, uncertainty=None):
        """
        Learn from veto experiences by adjusting Q-values
        
        Args:
            state: The state where veto occurred
            vetoed_action: The action that was vetoed
            alternative_action: The alternative action taken
            reward: The reward received after taking the alternative action
            uncertainty: Optional uncertainty estimate for the state
        """
        # Store the veto experience in replay buffer as a special transition
        if not hasattr(self, 'veto_buffer'):
            self.veto_buffer = []
        
        self.veto_buffer.append((state, vetoed_action, alternative_action, reward, uncertainty))
        
        # If we have enough veto experiences, update the Q-network to learn from them
        if len(self.veto_buffer) >= 10:
            self._train_from_veto_buffer()
        
    def _train_from_veto_buffer(self):
        """Train the agent using veto experiences"""
        if not hasattr(self, 'veto_buffer') or len(self.veto_buffer) < 5:
            return
            
        # Sample experiences from the veto buffer
        experiences = random.sample(self.veto_buffer, min(10, len(self.veto_buffer)))
        
        # For each veto experience, adjust Q-values
        for state, vetoed_action, alternative_action, reward, uncertainty in experiences:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get current Q-values
            self.model.eval()
            with torch.no_grad():
                if self.distributional:
                    q_values = self.model.get_q_values(state_tensor)
                else:
                    q_values = self.model(state_tensor)
                
            # Adjust target Q-values based on veto outcome
            target_q_values = q_values.clone().detach()
            
            # Use uncertainty to modulate adjustments
            uncertainty_factor = 1.0 if uncertainty is None else (1.0 - uncertainty)
            
            # If reward is positive, increase Q-value for alternative action
            if reward > 0:
                # Positive reward suggests alternative was good
                target_q_values[0, alternative_action] += min(reward, 0.5) * uncertainty_factor
                
                # Slightly decrease Q-value for vetoed action
                target_q_values[0, vetoed_action] -= 0.2 * uncertainty_factor
            elif reward < 0:
                # Negative reward suggests both actions might be suboptimal
                # Decrease both but vetoed action more
                target_q_values[0, vetoed_action] -= 0.3 * uncertainty_factor
                target_q_values[0, alternative_action] -= 0.1 * uncertainty_factor
            
            # Train the model on this adjusted data
            self.model.train()
            if self.distributional:
                # Handle distributional case
                # Implement distributional training logic here
                pass
            else:
                # Standard Q-learning update
                self.optimizer.zero_grad()
                loss = F.mse_loss(q_values, target_q_values)
                loss.backward()
                self.optimizer.step()
                
        # Clear buffer after training
        self.veto_buffer = self.veto_buffer[-5:]  # Keep recent experiences