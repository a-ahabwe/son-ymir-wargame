import torch
import numpy as np
import os
import json
from datetime import datetime
from src.ai.agent import RLAgent
from src.ai.uncertainty import UncertaintyEstimator
from src.ai.uncertainty_validation import UncertaintyValidator

class TrainingManager:
    """
    Manages the training process for RL agents with uncertainty validation.
    This version includes uncertainty validation after training.
    """
    def __init__(self, state_size, action_size, save_dir='data/trained_models', validate_uncertainty=True):
        self.state_size = state_size
        self.action_size = action_size
        self.save_dir = save_dir
        self.validate_uncertainty = validate_uncertainty
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Initialize agent
        self.agent = RLAgent(state_size, action_size)
        
        # Initialize uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(self.agent.model)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.loss_history = []
        
    def train_episode(self, env, max_steps=1000):
        """Train for one episode"""
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select action
            action, q_values = self.agent.select_action(state, self.uncertainty_estimator)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Train agent
            if len(self.agent.replay_buffer) >= self.agent.batch_size:
                loss = self.agent.train()
                if loss:
                    self.loss_history.append(loss)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
                
        # Record episode metrics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        return episode_reward, episode_length
        
    def train(self, env, num_episodes=100, save_frequency=10):
        """Train for multiple episodes"""
        for episode in range(num_episodes):
            episode_reward, episode_length = self.train_episode(env)
            
            # Print progress
            print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Length: {episode_length}")
            
            # Save model periodically
            if (episode + 1) % save_frequency == 0:
                self.save_agent(f"episode_{episode+1}")
                
        # Save final model
        model_path = self.save_agent("final")
        
        # Save training metrics
        self.save_metrics()
        
        # Run uncertainty validation if enabled
        if self.validate_uncertainty:
            print("\nRunning uncertainty validation...")
            validation_dir = os.path.join(self.save_dir, "uncertainty_validation")
            self.validate_model_uncertainty(env, validation_dir)
            
        return model_path
        
    def validate_model_uncertainty(self, env, validation_dir=None):
        """Validate uncertainty estimates of the trained model"""
        if validation_dir is None:
            validation_dir = os.path.join(self.save_dir, "uncertainty_validation")
            
        # Create uncertainty validator
        validator = UncertaintyValidator(
            self.agent.model,
            self.uncertainty_estimator,
            env,
            save_dir=validation_dir
        )
        
        # Run validation
        results = validator.run_all_validations()
        
        # Save validation results
        validation_results_path = os.path.join(validation_dir, "validation_results.json")
        with open(validation_results_path, 'w') as f:
            # Convert numpy values to Python native types
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2)
            
        print(f"Uncertainty validation results saved to {validation_results_path}")
        
        return results
        
    def _make_json_serializable(self, obj):
        """Convert numpy values to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
        
    def save_agent(self, name_suffix):
        """Save agent to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_dir}/agent_{name_suffix}_{timestamp}.pt"
        self.agent.save(filename)
        return filename
        
    def save_metrics(self):
        """Save training metrics to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_dir}/metrics_{timestamp}.json"
        
        metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "mean_length": np.mean(self.episode_lengths) if self.episode_lengths else 0,
            "final_epsilon": self.agent.epsilon
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        return filename
        
    def load_agent(self, path):
        """Load agent from file"""
        self.agent.load(path)
        # Re-initialize uncertainty estimator with loaded model
        self.uncertainty_estimator = UncertaintyEstimator(self.agent.model)