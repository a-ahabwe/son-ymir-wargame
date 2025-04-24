#!/usr/bin/env python3
"""
Script to train and evaluate a learning-based veto mechanism.
This demonstrates collecting training data and evaluating performance.
"""

import argparse
import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure the src package is in the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.game.environment import GameEnvironment
from src.ai.agent import RLAgent
from src.ai.uncertainty import UncertaintyEstimator
from src.veto.learning_risk_assessor import LearningRiskAssessor
from src.veto.learning_veto import LearningVetoMechanism
from src.veto.veto_mechanism import ThresholdVetoMechanism, UncertaintyVetoMechanism

def collect_training_data(num_episodes=100, max_steps=200, save_path=None):
    """
    Collect training data for the risk assessor by running the environment.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        save_path: Path to save collected data
    
    Returns:
        Collected experiences
    """
    # Set up environment and agent
    env = GameEnvironment(grid_size=50)
    agent = RLAgent(env.state_size, env.action_space_n)
    uncertainty_estimator = UncertaintyEstimator(agent.model)
    
    # Create a learning risk assessor to collect data
    risk_assessor = LearningRiskAssessor()
    
    # Track metrics
    total_rewards = []
    episode_lengths = []
    experiences = []
    
    print(f"Collecting training data over {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action
            action, q_values = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            
            # Extract features and record outcome
            features = risk_assessor.extract_features(state, action, q_values)
            was_risky = reward < 0
            
            # Store experience
            experiences.append((features, was_risky))
            
            # Update agent
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
        # Record episode metrics
        total_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, "
                 f"Avg Reward: {np.mean(total_rewards[-10:]):.2f}, "
                 f"Experiences: {len(experiences)}")
    
    # Save collected data if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save experiences
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump({
                'experiences': experiences,
                'metrics': {
                    'total_rewards': total_rewards,
                    'episode_lengths': episode_lengths
                }
            }, f)
            
        print(f"Training data saved to {save_path}")
    
    return experiences

def train_risk_model(experiences=None, data_path=None, model_type="random_forest", save_path=None):
    """
    Train a risk assessment model using collected experiences.
    
    Args:
        experiences: List of (features, label) tuples
        data_path: Path to load experiences if not provided
        model_type: Type of model to train
        save_path: Path to save trained model
    
    Returns:
        Trained risk assessor
    """
    # Load data if needed
    if experiences is None and data_path:
        import pickle
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            experiences = data['experiences']
    
    if not experiences:
        print("No training data provided")
        return None
    
    print(f"Training {model_type} risk model on {len(experiences)} experiences...")
    
    # Create risk assessor
    risk_assessor = LearningRiskAssessor(model_type=model_type)
    
    # Add experiences to buffer
    for features, label in experiences:
        risk_assessor.experience_buffer.append((features, label))
    
    # Train model
    success = risk_assessor.train_model()
    
    if success:
        # Print top feature importances
        importances = risk_assessor.get_feature_importances()
        if importances:
            print("\nTop Feature Importances:")
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            for name, importance in sorted_features[:10]:
                print(f"  {name}: {importance:.4f}")
        
        # Save model if requested
        if save_path:
            risk_assessor.save_model(save_path)
    
    return risk_assessor

def compare_veto_mechanisms(num_episodes=50, max_steps=200, risk_model_path=None):
    """
    Compare different veto mechanisms including the learning-based one.
    
    Args:
        num_episodes: Number of episodes to run for comparison
        max_steps: Maximum steps per episode
        risk_model_path: Path to pre-trained risk model
    """
    # Set up environment and agent
    env = GameEnvironment(grid_size=50)
    agent = RLAgent(env.state_size, env.action_space_n)
    uncertainty_estimator = UncertaintyEstimator(agent.model)
    
    # Create veto mechanisms to compare
    veto_mechanisms = {
        "No Veto": None,
        "Threshold": ThresholdVetoMechanism(threshold=0.7),
        "Uncertainty": UncertaintyVetoMechanism(uncertainty_estimator, uncertainty_threshold=0.5),
        "Learning": LearningVetoMechanism(
            uncertainty_estimator=uncertainty_estimator,
            risk_model_path=risk_model_path
        )
    }
    
    # Track metrics for each mechanism
    results = {name: {"rewards": [], "veto_counts": [], "successful_vetos": []} 
              for name in veto_mechanisms}
    
    print("Comparing veto mechanisms...")
    
    # Run comparison
    for name, veto in veto_mechanisms.items():
        print(f"\nTesting: {name}")
        
        for episode in range(num_episodes):
            # Reset environment
            state = env.reset()
            episode_reward = 0
            veto_count = 0
            successful_vetos = 0
            
            for step in range(max_steps):
                # Select action
                action, q_values = agent.select_action(state)
                
                # Check for veto if mechanism exists
                if veto:
                    veto_decision = veto.assess_action(state, action, q_values)
                    
                    if veto_decision.vetoed:
                        veto_count += 1
                        
                        # Choose alternative action
                        alternative_action = agent.select_safe_action(state, action)
                        
                        # Execute alternative action
                        next_state, reward, done, _ = env.step(alternative_action)
                        
                        # Record veto outcome for learning
                        veto.record_veto_decision(
                            state=state,
                            action=action,
                            vetoed=True,
                            alternative=alternative_action,
                            outcome=reward
                        )
                        
                        # Track successful vetos
                        if reward > 0:
                            successful_vetos += 1
                    else:
                        # Execute original action
                        next_state, reward, done, _ = env.step(action)
                else:
                    # No veto mechanism, just execute action
                    next_state, reward, done, _ = env.step(action)
                
                # Update agent
                agent.store_transition(state, action, reward, next_state, done)
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Record episode metrics
            results[name]["rewards"].append(episode_reward)
            results[name]["veto_counts"].append(veto_count)
            results[name]["successful_vetos"].append(successful_vetos)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(results[name]["rewards"][-10:])
                avg_vetos = np.mean(results[name]["veto_counts"][-10:])
                print(f"Episode {episode+1}/{num_episodes}, "
                     f"Avg Reward: {avg_reward:.2f}, Avg Vetos: {avg_vetos:.1f}")
    
    # Calculate summary statistics
    summary = {}
    for name, data in results.items():
        avg_reward = np.mean(data["rewards"])
        std_reward = np.std(data["rewards"])
        total_vetos = sum(data["veto_counts"])
        successful_rate = sum(data["successful_vetos"]) / max(1, total_vetos)
        
        summary[name] = {
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "total_vetos": total_vetos,
            "successful_veto_rate": successful_rate
        }
    
    # Print summary
    print("\nComparison Summary:")
    for name, stats in summary.items():
        print(f"\n{name}:")
        print(f"  Average Reward: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Total Vetos: {stats['total_vetos']}")
        if stats['total_vetos'] > 0:
            print(f"  Successful Veto Rate: {stats['successful_veto_rate']:.2f}")
    
    # Create visualizations
    plot_comparison_results(results, summary)
    
    return results, summary

def plot_comparison_results(results, summary):
    """Create visualizations of comparison results"""
    # Create plot directory
    os.makedirs("data/plots", exist_ok=True)
    
    # 1. Average reward comparison
    plt.figure(figsize=(10, 6))
    avg_rewards = [summary[name]["avg_reward"] for name in results]
    std_rewards = [summary[name]["std_reward"] for name in results]
    
    plt.bar(results.keys(), avg_rewards, yerr=std_rewards, capsize=10)
    plt.title("Average Reward by Veto Mechanism")
    plt.ylabel("Average Reward")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("data/plots/veto_comparison_rewards.png")
    plt.close()
    
    # 2. Veto count comparison
    mechanisms_with_veto = [name for name in results if name != "No Veto"]
    if mechanisms_with_veto:
        plt.figure(figsize=(10, 6))
        veto_counts = [summary[name]["total_vetos"] for name in mechanisms_with_veto]
        
        plt.bar(mechanisms_with_veto, veto_counts)
        plt.title("Total Veto Count by Mechanism")
        plt.ylabel("Veto Count")
        plt.grid(axis='y', alpha=0.3)
        plt.savefig("data/plots/veto_comparison_counts.png")
        plt.close()
        
        # 3. Successful veto rate comparison
        plt.figure(figsize=(10, 6))
        success_rates = [summary[name]["successful_veto_rate"] for name in mechanisms_with_veto]
        
        plt.bar(mechanisms_with_veto, success_rates)
        plt.title("Successful Veto Rate by Mechanism")
        plt.ylabel("Success Rate")
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig("data/plots/veto_comparison_success_rates.png")
        plt.close()
    
    # 4. Reward over time
    plt.figure(figsize=(12, 6))
    
    for name, data in results.items():
        # Calculate moving average
        window_size = min(10, len(data["rewards"]))
        moving_avg = [np.mean(data["rewards"][max(0, i-window_size):i+1]) 
                     for i in range(len(data["rewards"]))]
        
        plt.plot(moving_avg, label=name)
    
    plt.title("Reward Over Time by Veto Mechanism")
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("data/plots/veto_comparison_learning_curve.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate learning-based veto')
    parser.add_argument('--collect_data', action='store_true', help='Collect training data')
    parser.add_argument('--train_model', action='store_true', help='Train risk model')
    parser.add_argument('--compare', action='store_true', help='Compare veto mechanisms')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--model_type', type=str, default='random_forest',
                      choices=['random_forest', 'gradient_boosting', 'neural_network'],
                      help='Type of machine learning model to use')
    parser.add_argument('--data_path', type=str, default='data/risk_model/training_data.pkl',
                      help='Path to save/load training data')
    parser.add_argument('--model_path', type=str, default='data/risk_model/risk_model.joblib',
                      help='Path to save/load risk model')
    
    args = parser.parse_args()
    
    # Check if any action is specified
    if not (args.collect_data or args.train_model or args.compare):
        # Default to running all steps
        args.collect_data = True
        args.train_model = True
        args.compare = True
        
    # Create output directories
    os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Step 1: Collect training data
    experiences = None
    if args.collect_data:
        experiences = collect_training_data(
            num_episodes=args.episodes,
            save_path=args.data_path
        )
    
    # Step 2: Train risk model
    if args.train_model:
        train_risk_model(
            experiences=experiences,
            data_path=args.data_path if not experiences else None,
            model_type=args.model_type,
            save_path=args.model_path
        )
    
    # Step 3: Compare veto mechanisms
    if args.compare:
        compare_veto_mechanisms(
            num_episodes=args.episodes,
            risk_model_path=args.model_path
        )
    
    print("\nAll operations completed successfully.")

if __name__ == "__main__":
    main()