import torch
import numpy as np
import os
import json
from datetime import datetime
from src.ai.agent import RLAgent

class TrainingManager:
    """Manages the training process for RL agents"""
    def __init__(self, state_size, action_size, save_dir='data/trained_models'):
        self.state_size = state_size
        self.action_size = action_size
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Initialize agent
        self.agent = RLAgent(state_size, action_size)
        
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
            action, _ = self.agent.select_action(state)
            
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
        self.save_agent("final")
        
        # Save training metrics
        self.save_metrics()
        
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
            json.dump(metrics, f)
            
        return filename
        
    def load_agent(self, path):
        """Load agent from file"""
        self.agent.load(path)