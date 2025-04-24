"""
Selection bias mitigation module.
Provides methods to address the selection bias in veto learning.
"""

import numpy as np
import torch
import time
import random
from collections import defaultdict, deque
from src.game.state import GameState

class CounterfactualEstimator:
    """
    Estimator for counterfactual outcomes of vetoed actions.
    Addresses the selection bias in veto learning by estimating what would
    have happened if vetoed actions had been taken.
    """
    def __init__(self, model=None, environment_class=None, num_simulations=10):
        """
        Initialize counterfactual estimator
        
        Args:
            model: Model for reward prediction (optional)
            environment_class: Class of the environment for simulation (optional)
            num_simulations: Number of simulations for Monte Carlo estimation
        """
        self.model = model
        self.environment_class = environment_class
        self.num_simulations = num_simulations
        
        # Experience buffer for training reward predictor
        self.experience_buffer = deque(maxlen=10000)
        
        # Cache for counterfactual estimates to avoid redundant computation
        self.estimate_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Statistics
        self.stats = defaultdict(float)
    
    def estimate_counterfactual(self, state, action, method='combined'):
        """
        Estimate the counterfactual outcome of a vetoed action
        
        Args:
            state: State at which action was vetoed
            action: Vetoed action
            method: Estimation method ('model', 'simulation', 'combined')
            
        Returns:
            (estimated_reward, confidence, explanation): Counterfactual estimate
        """
        # Check cache first
        cache_key = self._create_cache_key(state, action, method)
        if cache_key in self.estimate_cache:
            self.cache_hits += 1
            return self.estimate_cache[cache_key]
            
        self.cache_misses += 1
        
        # Use different estimation methods
        estimates = []
        confidences = []
        explanations = []
        
        # Model-based estimation
        if method in ['model', 'combined'] and self.model is not None:
            reward, conf, expl = self._model_based_estimation(state, action)
            estimates.append(reward)
            confidences.append(conf)
            explanations.append(expl)
            
        # Simulation-based estimation
        if method in ['simulation', 'combined'] and self.environment_class is not None:
            reward, conf, expl = self._simulation_based_estimation(state, action)
            estimates.append(reward)
            confidences.append(conf)
            explanations.append(expl)
            
        # If no applicable methods, use simple heuristic estimate
        if not estimates:
            reward, conf, expl = self._heuristic_estimation(state, action)
            estimates.append(reward)
            confidences.append(conf)
            explanations.append(expl)
            
        # Combine estimates weighted by confidence
        total_confidence = sum(confidences)
        if total_confidence > 0:
            estimated_reward = sum(e * c / total_confidence for e, c in zip(estimates, confidences))
            confidence = sum(confidences) / len(confidences)  # Average confidence
        else:
            estimated_reward = sum(estimates) / len(estimates)  # Simple average
            confidence = 0.5  # Default medium confidence
        
        # Create combined explanation
        if len(explanations) == 1:
            explanation = explanations[0]
        else:
            explanation = "Combined estimate: " + "; ".join(explanations)
            
        # Store in cache
        result = (estimated_reward, confidence, explanation)
        self.estimate_cache[cache_key] = result
        
        return result
    
    def _model_based_estimation(self, state, action):
        """
        Estimate counterfactual using a reward prediction model
        
        Args:
            state: State at which action was vetoed
            action: Vetoed action
            
        Returns:
            (estimated_reward, confidence, explanation): Estimate
        """
        # Convert to appropriate format
        if isinstance(state, GameState):
            state_tensor = state.to_tensor().unsqueeze(0)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
        # Create one-hot encoding of action
        action_tensor = torch.zeros(1, self.model.action_size)
        action_tensor[0, action] = 1.0
        
        # Predict reward using model
        with torch.no_grad():
            input_tensor = torch.cat([state_tensor, action_tensor], dim=1)
            predicted_reward = self.model(input_tensor).item()
            
        # Estimate confidence based on training data
        state_features = state.raw if isinstance(state, GameState) else state
        confidence = self._estimate_prediction_confidence(state_features, action)
        
        # Create explanation
        explanation = f"Model prediction: {predicted_reward:.2f} (confidence: {confidence:.2f})"
        
        return predicted_reward, confidence, explanation
    
    def _simulation_based_estimation(self, state, action):
        """
        Estimate counterfactual using environment simulations
        
        Args:
            state: State at which action was vetoed
            action: Vetoed action
            
        Returns:
            (estimated_reward, confidence, explanation): Estimate
        """
        # Run multiple simulations
        rewards = []
        for _ in range(self.num_simulations):
            # Create environment instance
            env = self.environment_class()
            
            # Set environment to match the given state
            self._set_environment_state(env, state)
            
            # Take the action
            _, reward, done, _ = env.step(action)
            rewards.append(reward)
        
        # Calculate statistics
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        # Confidence inversely proportional to standard deviation
        confidence = 1.0 / (1.0 + std_reward * 3.0)  # Range: 0 to 1
        
        # Create explanation
        explanation = f"Simulation: avg={avg_reward:.2f}, std={std_reward:.2f}"
        
        return avg_reward, confidence, explanation
    
    def _heuristic_estimation(self, state, action):
        """
        Simple heuristic estimation when other methods not available
        
        Args:
            state: State at which action was vetoed
            action: Vetoed action
            
        Returns:
            (estimated_reward, confidence, explanation): Estimate
        """
        # Convert to GameState if needed
        if not isinstance(state, GameState):
            try:
                state = GameState(raw_state=state)
            except:
                # Fall back to very simple heuristic
                return -0.2, 0.3, "Default heuristic estimate"
        
        # Simple heuristic based on agent stats and action type
        health = state.health
        ammo = state.ammo
        shields = state.shields
        
        # Default slightly negative expectation for vetoed actions
        estimated_reward = -0.1
        
        # Adjust based on action type
        if action <= 3:  # Movement actions
            # Movement generally safe but low reward
            estimated_reward = 0.1
        elif action <= 7:  # Combat actions
            # Combat risky with low ammo
            if ammo < 0.2:
                estimated_reward = -0.5
            else:
                # Otherwise can be positive
                estimated_reward = 0.3
        else:  # Special actions
            # Special actions with low health often risky
            if health < 0.3:
                estimated_reward = -0.3
            else:
                estimated_reward = 0.2
        
        # Low confidence in heuristic
        confidence = 0.3
        
        # Create explanation
        explanation = f"Heuristic estimate: {estimated_reward:.2f} (low confidence)"
        
        return estimated_reward, confidence, explanation
    
    def _set_environment_state(self, env, state):
        """Set environment to match the given state (implementation depends on environment)"""
        # This is a simplified implementation - a full implementation would need
        # to properly set all aspects of the environment state
        
        # Reset environment
        env.reset()
        
        # Convert to GameState if needed
        if not isinstance(state, GameState):
            state = GameState(raw_state=state)
        
        # Set agent position and stats
        if hasattr(env, 'agent_pos'):
            # For now, just keep agent at current position
            pass
            
        if hasattr(env, 'agent_health'):
            env.agent_health = state.health * 100  # Scale from [0,1] to [0,100]
            
        if hasattr(env, 'agent_ammo'):
            env.agent_ammo = state.ammo * 30  # Assuming max ammo is 30
            
        if hasattr(env, 'agent_shields'):
            env.agent_shields = int(state.shields * 5)  # Assuming max shields is 5
    
    def _estimate_prediction_confidence(self, state_features, action):
        """
        Estimate confidence in the prediction based on similar experiences
        
        Args:
            state_features: Features of the state
            action: Action
            
        Returns:
            Confidence score in [0,1]
        """
        # Simple baseline confidence
        if not self.experience_buffer:
            return 0.5
            
        # Count experiences with similar state and same action
        similarities = []
        for exp in self.experience_buffer:
            exp_state, exp_action, _, _ = exp
            
            # Skip different actions
            if exp_action != action:
                continue
                
            # Calculate state similarity
            similarity = self._calculate_similarity(state_features, exp_state)
            similarities.append(similarity)
        
        # Get average similarity of top 5 most similar experiences
        top_similarities = sorted(similarities, reverse=True)[:5]
        
        if not top_similarities:
            return 0.3  # Low confidence if no similar experiences
            
        avg_similarity = sum(top_similarities) / len(top_similarities)
        
        # Map similarity to confidence
        confidence = 0.3 + avg_similarity * 0.6  # Range: 0.3 to 0.9
        
        return confidence
    
    def _calculate_similarity(self, state1, state2):
        """
        Calculate similarity between two states
        
        Args:
            state1: First state features
            state2: Second state features
            
        Returns:
            Similarity score in [0,1]
        """
        # Ensure states are numpy arrays
        s1 = np.array(state1)
        s2 = np.array(state2)
        
        # Handle different lengths
        min_len = min(len(s1), len(s2))
        s1 = s1[:min_len]
        s2 = s2[:min_len]
        
        # Euclidean distance
        distance = np.sqrt(np.sum((s1 - s2) ** 2))
        
        # Convert to similarity (1.0 means identical)
        similarity = 1.0 / (1.0 + distance)
        
        return similarity
    
    def record_experience(self, state, action, reward, next_state=None):
        """
        Record an experience to improve future counterfactual estimates
        
        Args:
            state: State
            action: Action taken
            reward: Reward received
            next_state: Resulting state (optional)
        """
        # Convert to proper format
        if isinstance(state, GameState):
            state = state.raw
            
        if isinstance(next_state, GameState):
            next_state = next_state.raw
            
        # Add to experience buffer
        self.experience_buffer.append((state, action, reward, next_state))
        
        # Update statistics
        action_key = f"action_{action}"
        self.stats[f"{action_key}_count"] += 1
        self.stats[f"{action_key}_reward"] = (
            (self.stats[f"{action_key}_reward"] * (self.stats[f"{action_key}_count"] - 1) + reward) / 
            self.stats[f"{action_key}_count"]
        )
    
    def train_reward_model(self):
        """Train the reward prediction model on collected experiences"""
        if self.model is None or len(self.experience_buffer) < 100:
            return False
            
        # Extract training data
        states = []
        actions = []
        rewards = []
        
        for state, action, reward, _ in self.experience_buffer:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
        # Convert to tensors
        state_tensor = torch.FloatTensor(states)
        
        # Create one-hot encoding of actions
        action_tensor = torch.zeros(len(actions), self.model.action_size)
        for i, a in enumerate(actions):
            action_tensor[i, a] = 1.0
            
        # Create input by concatenating state and action
        input_tensor = torch.cat([state_tensor, action_tensor], dim=1)
        
        # Target rewards
        reward_tensor = torch.FloatTensor(rewards)
        
        # Train model with simple MSE loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Simple training loop
        num_epochs = 50
        batch_size = min(32, len(states))
        
        for epoch in range(num_epochs):
            # Shuffle data
            indices = list(range(len(states)))
            random.shuffle(indices)
            
            # Train in batches
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                
                # Get batch
                batch_input = input_tensor[batch_indices]
                batch_target = reward_tensor[batch_indices]
                
                # Forward pass
                optimizer.zero_grad()
                prediction = self.model(batch_input)
                
                # Calculate loss
                loss = criterion(prediction, batch_target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
        
        # Evaluate on training data
        with torch.no_grad():
            predictions = self.model(input_tensor)
            mse = criterion(predictions, reward_tensor).item()
            
        return {
            'mse': mse,
            'num_samples': len(states),
            'num_epochs': num_epochs
        }
    
    def _create_cache_key(self, state, action, method):
        """Create cache key for counterfactual estimates"""
        # Use state values, action and method as key
        if isinstance(state, GameState):
            state_array = state.raw
        else:
            state_array = state
            
        # Use significant state features for cache key
        # This is a compromise between precision and cache efficiency
        if len(state_array) > 20:
            # Use agent stats and a subset of state features
            if isinstance(state, GameState):
                key_values = [state.health, state.ammo, state.shields]
            else:
                # Assume last 3 values are agent stats
                key_values = list(state_array[-3:])
                
            # Add some additional state features
            sample_indices = [i for i in range(0, min(len(state_array) - 3, 50), 10)]
            key_values.extend([state_array[i] for i in sample_indices])
        else:
            # For small states, use full state
            key_values = list(state_array)
            
        # Round values for cache efficiency
        key_values = [float(np.round(v, 3)) for v in key_values]
        
        # Create key from action, method and state values
        return (action, method, tuple(key_values))
    
    def reset_cache(self):
        """Clear the counterfactual cache"""
        self.estimate_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_stats(self):
        """Get estimator statistics"""
        stats = dict(self.stats)
        stats.update({
            'num_experiences': len(self.experience_buffer),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_ratio': self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        })
        return stats


class OffPolicyEvaluator:
    """
    Evaluator for off-policy learning from veto decisions.
    Enables learning from both taken and vetoed actions.
    """
    def __init__(self, counterfactual_estimator=None):
        """
        Initialize off-policy evaluator
        
        Args:
            counterfactual_estimator: Estimator for counterfactual outcomes
        """
        self.counterfactual_estimator = counterfactual_estimator or CounterfactualEstimator()
        
        # Statistics for off-policy learning
        self.stats = {
            'total_decisions': 0,
            'vetoed_decisions': 0,
            'approved_decisions': 0,
            'vetoed_estimated_reward': 0,
            'approved_actual_reward': 0
        }
    
    def process_veto_decision(self, state, action, vetoed, alternative, outcome):
        """
        Process a veto decision for off-policy learning
        
        Args:
            state: State where veto decision was made
            action: Original action
            vetoed: Whether the action was vetoed
            alternative: Alternative action taken if vetoed
            outcome: Actual outcome observed
            
        Returns:
            Dictionary with processed data for learning
        """
        # Extract actual reward from outcome
        if isinstance(outcome, tuple):
            actual_reward = outcome[0]
        else:
            actual_reward = outcome
            
        # Update statistics
        self.stats['total_decisions'] += 1
        
        if vetoed:
            self.stats['vetoed_decisions'] += 1
            
            # Estimate counterfactual outcome for the vetoed action
            estimated_reward, confidence, explanation = (
                self.counterfactual_estimator.estimate_counterfactual(state, action)
            )
            
            # Update statistics
            self.stats['vetoed_estimated_reward'] += estimated_reward
            
            # Create learning data for the vetoed action
            original_data = {
                'state': state,
                'action': action,
                'reward': estimated_reward,
                'is_counterfactual': True,
                'confidence': confidence,
                'explanation': explanation
            }
            
            # Record the actual taken action and outcome
            if alternative is not None:
                self.counterfactual_estimator.record_experience(
                    state, alternative, actual_reward
                )
                
                # Create learning data for the alternative action
                alternative_data = {
                    'state': state,
                    'action': alternative,
                    'reward': actual_reward,
                    'is_counterfactual': False,
                    'confidence': 1.0,
                    'explanation': "Actual observed outcome"
                }
                
                return {
                    'original': original_data,
                    'alternative': alternative_data
                }
            else:
                return {
                    'original': original_data
                }
        else:
            # Action wasn't vetoed, record the actual outcome
            self.stats['approved_decisions'] += 1
            self.stats['approved_actual_reward'] += actual_reward
            
            # Record experience for future counterfactual estimates
            self.counterfactual_estimator.record_experience(
                state, action, actual_reward
            )
            
            # Create learning data
            data = {
                'state': state,
                'action': action,
                'reward': actual_reward,
                'is_counterfactual': False,
                'confidence': 1.0,
                'explanation': "Actual observed outcome"
            }
            
            return {
                'original': data
            }
    
    def get_importance_weights(self, veto_probability, is_vetoed):
        """
        Calculate importance sampling weights to correct for selection bias
        
        Args:
            veto_probability: Probability of veto for the action
            is_vetoed: Whether the action was actually vetoed
            
        Returns:
            Importance weight for the sample
        """
        # Importance weight calculation:
        # For vetoed actions: 1 / veto_probability
        # For non-vetoed actions: 1 / (1 - veto_probability)
        if is_vetoed:
            # Avoid division by zero
            prob = max(0.01, min(0.99, veto_probability))
            return 1.0 / prob
        else:
            # Avoid division by zero
            prob = max(0.01, min(0.99, veto_probability))
            return 1.0 / (1.0 - prob)
    
    def get_average_rewards(self):
        """Calculate average rewards for vetoed and approved actions"""
        vetoed_avg = (
            self.stats['vetoed_estimated_reward'] / max(1, self.stats['vetoed_decisions'])
        )
        
        approved_avg = (
            self.stats['approved_actual_reward'] / max(1, self.stats['approved_decisions'])
        )
        
        return {
            'vetoed_avg_reward': vetoed_avg,
            'approved_avg_reward': approved_avg,
            'veto_count': self.stats['vetoed_decisions'],
            'approved_count': self.stats['approved_decisions']
        }
    
    def get_stats(self):
        """Get evaluator statistics"""
        stats = self.stats.copy()
        
        # Add counterfactual estimator stats
        counterfactual_stats = self.counterfactual_estimator.get_stats()
        stats['counterfactual'] = counterfactual_stats
        
        # Add average rewards
        avg_rewards = self.get_average_rewards()
        stats.update(avg_rewards)
        
        return stats