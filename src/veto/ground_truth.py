"""
Ground truth module for veto decisions.
Provides methods to establish when vetoes are truly justified.
"""

import numpy as np
import time
from collections import defaultdict
from src.game.state import GameState

class GroundTruthOracle:
    """
    Oracle for establishing ground truth on when vetoes are justified.
    Uses multiple methods to evaluate decision quality and determine
    true risk levels for actions.
    """
    def __init__(self, environment_class, num_simulations=10, simulation_horizon=20):
        self.environment_class = environment_class  # Class, not instance
        self.num_simulations = num_simulations
        self.simulation_horizon = simulation_horizon
        
        # Cache for simulation results to avoid redundant computation
        self.simulation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance metrics
        self.metrics = {
            'simulation_time': 0,
            'total_simulations': 0,
            'actions_evaluated': 0
        }
        
        # Action outcome statistics for different conditions
        self.outcome_stats = defaultdict(lambda: {
            'count': 0,
            'positive_outcomes': 0,
            'negative_outcomes': 0,
            'avg_reward': 0,
            'avg_regret': 0
        })
    
    def should_veto(self, state, action, q_values=None, threshold=0.5):
        """
        Determine if an action should be vetoed based on ground truth simulation
        
        Args:
            state: GameState object or raw state array
            action: Action to evaluate
            q_values: Optional Q-values for reference
            threshold: Risk threshold for veto decision
            
        Returns:
            (should_veto, confidence, explanation): Decision tuple
        """
        # Convert to GameState if needed
        if not isinstance(state, GameState):
            state = GameState(raw_state=state)
            
        # Run decision analysis
        risk_score, metrics, explanation = self.evaluate_action_risk(state, action)
        
        # Make veto decision
        should_veto = risk_score > threshold
        confidence = metrics.get('confidence', 0.7)
        
        return should_veto, confidence, explanation
    
    def evaluate_action_risk(self, state, action):
        """
        Evaluate the risk of an action using multiple methods
        
        Args:
            state: GameState object
            action: Action to evaluate
            
        Returns:
            (risk_score, metrics, explanation): Evaluation results
        """
        # Check cache first
        cache_key = self._create_cache_key(state, action)
        if cache_key in self.simulation_cache:
            self.cache_hits += 1
            return self.simulation_cache[cache_key]
        
        self.cache_misses += 1
        self.metrics['actions_evaluated'] += 1
        
        # Combine multiple evaluation methods
        start_time = time.time()
        
        # 1. Monte Carlo simulation
        mc_risk, mc_metrics, mc_explanation = self._monte_carlo_evaluation(state, action)
        
        # 2. Outcome regression (predict reward directly)
        regression_risk, reg_metrics, reg_explanation = self._outcome_regression(state, action)
        
        # 3. Counterfactual analysis (compare with best action)
        cf_risk, cf_metrics, cf_explanation = self._counterfactual_analysis(state, action)
        
        # Track simulation time
        sim_time = time.time() - start_time
        self.metrics['simulation_time'] += sim_time
        
        # Combine evaluations (weighted average)
        # Weight based on confidence of each method
        mc_weight = mc_metrics.get('confidence', 0.5)
        reg_weight = reg_metrics.get('confidence', 0.2)
        cf_weight = cf_metrics.get('confidence', 0.3)
        
        total_weight = mc_weight + reg_weight + cf_weight
        risk_score = (
            (mc_risk * mc_weight) + 
            (regression_risk * reg_weight) + 
            (cf_risk * cf_weight)
        ) / total_weight
        
        # Create combined metrics
        metrics = {
            'mc_risk': mc_risk,
            'regression_risk': regression_risk,
            'counterfactual_risk': cf_risk,
            'simulation_time': sim_time,
            'confidence': (mc_metrics['confidence'] + reg_metrics['confidence'] + cf_metrics['confidence']) / 3
        }
        
        # Create combined explanation
        explanation = f"Action risk: {risk_score:.2f}\n"
        explanation += f"• Monte Carlo: {mc_explanation}\n"
        explanation += f"• Outcome prediction: {reg_explanation}\n"
        explanation += f"• Counterfactual: {cf_explanation}"
        
        # Cache result
        result = (risk_score, metrics, explanation)
        self.simulation_cache[cache_key] = result
        
        return result
    
    def _monte_carlo_evaluation(self, state, action):
        """
        Evaluate action using Monte Carlo simulations
        
        Args:
            state: GameState object
            action: Action to evaluate
            
        Returns:
            (risk_score, metrics, explanation): Evaluation results
        """
        # Create environment instances for simulation
        rewards = []
        negative_outcomes = 0
        fatal_outcomes = 0
        
        # Run multiple simulations
        for i in range(self.num_simulations):
            # Create new environment
            env = self.environment_class()
            
            # Set environment to match the given state
            self._set_environment_state(env, state)
            
            # Take the action
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            
            # Track outcomes
            if reward < 0:
                negative_outcomes += 1
            if done and reward < 0:
                fatal_outcomes += 1
                
            # Continue simulation for a few more steps to see long-term impact
            if not done:
                cumulative_reward = reward
                for _ in range(self.simulation_horizon - 1):
                    # Take random actions for remaining steps
                    random_action = np.random.randint(0, env.action_space_n)
                    next_state, step_reward, done, _ = env.step(random_action)
                    cumulative_reward += step_reward
                    if done:
                        break
                
                # Add longer-term reward to consideration (discounted)
                rewards[-1] += cumulative_reward * 0.5  # Discount future rewards
        
        # Calculate metrics
        avg_reward = np.mean(rewards)
        negative_rate = negative_outcomes / self.num_simulations
        fatal_rate = fatal_outcomes / self.num_simulations
        reward_std = np.std(rewards)
        
        # Calculate risk score
        # High risk if: many negative outcomes OR fatal outcomes OR high variance
        risk_score = (negative_rate * 0.5) + (fatal_rate * 0.8) + min(1.0, reward_std * 0.3)
        
        # Reduce risk if average reward is positive
        if avg_reward > 0:
            risk_score = max(0, risk_score - 0.3)
            
        # Increase risk if average reward is very negative
        if avg_reward < -0.5:
            risk_score = min(1.0, risk_score + 0.3)
            
        # Normalize risk score
        risk_score = min(1.0, max(0.0, risk_score))
        
        # Create metrics dictionary
        metrics = {
            'avg_reward': avg_reward,
            'negative_rate': negative_rate,
            'fatal_rate': fatal_rate,
            'reward_std': reward_std,
            'confidence': 0.7  # Confidence in MC evaluation
        }
        
        # Create explanation
        explanation = (
            f"Simulation: avg reward={avg_reward:.2f}, "
            f"negative outcomes={negative_rate*100:.0f}%, "
            f"fatal outcomes={fatal_rate*100:.0f}%"
        )
        
        # Track statistics
        self.metrics['total_simulations'] += self.num_simulations
        
        # Update outcome statistics
        key = f"action_{action}"
        stats = self.outcome_stats[key]
        stats['count'] += 1
        stats['avg_reward'] = ((stats['avg_reward'] * (stats['count'] - 1)) + avg_reward) / stats['count']
        stats['positive_outcomes'] = ((stats['positive_outcomes'] * (stats['count'] - 1)) + 
                                   (self.num_simulations - negative_outcomes)) / stats['count']
        stats['negative_outcomes'] = ((stats['negative_outcomes'] * (stats['count'] - 1)) + 
                                    negative_outcomes) / stats['count']
        
        return risk_score, metrics, explanation
    
    def _outcome_regression(self, state, action):
        """
        Predict action outcome directly using regression
        
        Args:
            state: GameState object
            action: Action to evaluate
            
        Returns:
            (risk_score, metrics, explanation): Evaluation results
        """
        # For the initial implementation, use a simplified outcome prediction
        # based on state and action - this would be replaced with an actual
        # trained regression model in a full implementation
        
        # For now, use state statistics as a proxy
        health = state.health
        ammo = state.ammo
        shields = state.shields
        
        # Predict expected reward based on action type and resources
        expected_reward = 0.5  # Default neutral prediction
        
        # Adjust based on action type
        if action <= 3:  # Movement
            expected_reward = 0.2  # Generally safe but low reward
        elif action <= 7:  # Combat
            # Combat reward depends on ammo
            if ammo < 0.2:
                expected_reward = -0.3  # Negative if low ammo
            else:
                expected_reward = 0.8  # High if sufficient ammo
        else:  # Special actions
            expected_reward = 0.4  # Moderate reward
            
        # Adjust based on health
        if health < 0.3:
            expected_reward -= 0.3  # Higher risk with low health
            
        # Predict probability of negative outcome
        neg_prob = max(0, min(1, 0.5 - expected_reward))
        
        # Calculate risk score based on prediction
        risk_score = neg_prob * 0.7 + (1 - expected_reward/2) * 0.3
        
        # Normalize risk score
        risk_score = min(1.0, max(0.0, risk_score))
        
        # Create metrics
        metrics = {
            'expected_reward': expected_reward,
            'neg_prob': neg_prob,
            'confidence': 0.5  # Lower confidence than MC simulation
        }
        
        # Create explanation
        explanation = f"Predicted reward={expected_reward:.2f}, negative probability={neg_prob:.2f}"
        
        return risk_score, metrics, explanation
    
    def _counterfactual_analysis(self, state, action):
        """
        Compare with counterfactual actions (what if we did something else?)
        
        Args:
            state: GameState object
            action: Action to evaluate
            
        Returns:
            (risk_score, metrics, explanation): Evaluation results
        """
        # First evaluate the proposed action
        env = self.environment_class()
        self._set_environment_state(env, state)
        
        # Take the action and get reward
        _, action_reward, action_done, _ = env.step(action)
        
        # Evaluate alternative actions to establish baseline for comparison
        rewards = []
        for alt_action in range(env.action_space_n):
            if alt_action == action:
                continue  # Skip the action we're evaluating
                
            # Create new environment
            alt_env = self.environment_class()
            self._set_environment_state(alt_env, state)
            
            # Take alternative action
            _, alt_reward, _, _ = alt_env.step(alt_action)
            rewards.append(alt_reward)
        
        # Calculate regret: difference between this action and the best alternative
        best_alt_reward = max(rewards) if rewards else 0
        regret = max(0, best_alt_reward - action_reward)
        
        # Calculate metrics
        avg_alt_reward = np.mean(rewards) if rewards else 0
        reward_percentile = sum(1 for r in rewards if r <= action_reward) / max(1, len(rewards))
        
        # Calculate risk score based on regret and relative performance
        risk_score = (regret * 0.6) + ((1 - reward_percentile) * 0.4)
        
        # Normalize risk score
        risk_score = min(1.0, max(0.0, risk_score))
        
        # Create metrics
        metrics = {
            'action_reward': action_reward,
            'best_alt_reward': best_alt_reward,
            'avg_alt_reward': avg_alt_reward,
            'regret': regret,
            'reward_percentile': reward_percentile,
            'confidence': 0.6  # Medium confidence
        }
        
        # Create explanation
        explanation = (
            f"Regret={regret:.2f}, reward percentile={reward_percentile*100:.0f}%, "
            f"best alternative reward={best_alt_reward:.2f}"
        )
        
        # Update outcome statistics
        key = f"action_{action}"
        stats = self.outcome_stats[key]
        stats['avg_regret'] = ((stats['avg_regret'] * stats['count']) + regret) / (stats['count'] + 1)
        
        return risk_score, metrics, explanation
    
    def _set_environment_state(self, env, state):
        """Set environment to match the given state (implementation depends on environment)"""
        # This is a simplified implementation - a full implementation would need
        # to properly set all aspects of the environment state
        
        # Reset environment
        env.reset()
        
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
    
    def _create_cache_key(self, state, action):
        """Create cache key for simulation results"""
        # Use state values and action as key
        if isinstance(state, GameState):
            state_array = state.raw
        else:
            state_array = state
            
        # Use first few and last few state values plus key statistics for cache key
        # This is a compromise between precision and cache efficiency
        key_indices = list(range(min(10, len(state_array)))) + list(range(max(0, len(state_array) - 3), len(state_array)))
        key_values = [state_array[i] for i in key_indices]
        
        # Create key from action and key state values
        return (action, tuple(np.round(key_values, 4)))
    
    def reset_cache(self):
        """Clear the simulation cache"""
        self.simulation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_metrics(self):
        """Get oracle performance metrics"""
        metrics = self.metrics.copy()
        metrics.update({
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_ratio': self.cache_hits / max(1, (self.cache_hits + self.cache_misses))
        })
        return metrics
    
    def get_action_statistics(self):
        """Get action outcome statistics"""
        return dict(self.outcome_stats)