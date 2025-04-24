#!/usr/bin/env python3
"""
Benchmark script to compare different agent and veto mechanism approaches.
This allows quantitative comparison between our complex approaches and simple baselines.
"""

import sys
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Ensure the src package is in the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.game.environment import GameEnvironment
from src.ai.agent import RLAgent
from src.ai.baseline_agents import RandomAgent, RuleBasedAgent, SimpleQLearningAgent
from src.ai.uncertainty import UncertaintyEstimator
from src.veto.veto_mechanism import VetoMechanism, ThresholdVetoMechanism, UncertaintyVetoMechanism
from src.veto.baseline_veto import RandomVetoMechanism, FixedRulesVetoMechanism, QValueThresholdVeto, HistoricalPerformanceVeto

def run_benchmark(agent, veto_mechanism, env, num_episodes=20, max_steps=200):
    """Run benchmark for a specific agent and veto mechanism combination"""
    episode_rewards = []
    episode_lengths = []
    veto_counts = []
    successful_vetos = []  # Vetoes that led to positive rewards
    
    print(f"Running benchmark with {agent.__class__.__name__} and {veto_mechanism.__class__.__name__}")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_veto_count = 0
        episode_successful_vetos = 0
        
        for step in range(max_steps):
            # Select action
            action, q_values = agent.select_action(state)
            
            # Check for veto
            veto_decision = veto_mechanism.assess_action(state, action, q_values)
            
            if veto_decision.vetoed:
                episode_veto_count += 1
                # Use agent's safe action selection or simple rule-based alternative
                if hasattr(agent, 'select_safe_action'):
                    alternative_action = agent.select_safe_action(state, action)
                else:
                    # Simple rule: just choose a different action
                    alternative_action = (action + 1) % env.action_space_n
                
                # Execute alternative action
                next_state, reward, done, _ = env.step(alternative_action)
                
                # Was this a successful veto?
                if reward > 0:
                    episode_successful_vetos += 1
                
                # Record veto decision
                veto_mechanism.record_veto_decision(
                    state=state,
                    action=action,
                    vetoed=True,
                    alternative=alternative_action,
                    outcome=(reward, done)
                )
            else:
                # Execute original action
                next_state, reward, done, _ = env.step(action)
            
            # Store transition for learning agents
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent if applicable
            if hasattr(agent, 'train'):
                agent.train()
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        veto_counts.append(episode_veto_count)
        successful_vetos.append(episode_successful_vetos)
        
        # Print progress
        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Length: {episode_length}, "
              f"Vetos: {episode_veto_count}, Successful: {episode_successful_vetos}")
    
    # Calculate overall metrics
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    total_vetos = sum(veto_counts)
    successful_veto_ratio = sum(successful_vetos) / max(1, total_vetos)
    
    print(f"Benchmark results: Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}, "
          f"Veto Count: {total_vetos}, Successful Veto Ratio: {successful_veto_ratio:.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'veto_counts': veto_counts,
        'successful_vetos': successful_vetos,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'total_vetos': total_vetos,
        'successful_veto_ratio': successful_veto_ratio
    }

def main():
    parser = argparse.ArgumentParser(description='Benchmark various agent and veto combinations')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes per benchmark')
    parser.add_argument('--steps', type=int, default=200, help='Maximum steps per episode')
    parser.add_argument('--save_dir', type=str, default='data/benchmark_results', help='Directory to save results')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained DQN model')
    parser.add_argument('--grid_size', type=int, default=50, help='Size of game grid')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create environment
    env = GameEnvironment(grid_size=args.grid_size)
    
    # Create agents
    agents = {
        'Random': RandomAgent(env.state_size, env.action_space_n),
        'RuleBased': RuleBasedAgent(env.state_size, env.action_space_n),
        'SimpleQLearning': SimpleQLearningAgent(env.state_size, env.action_space_n),
        'DQN': RLAgent(env.state_size, env.action_space_n)
    }
    
    # Load DQN model if available
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        agents['DQN'].load(args.model_path)
    
    # Create uncertainty estimator for DQN
    uncertainty_estimator = UncertaintyEstimator(agents['DQN'].model)
    
    # Create veto mechanisms
    veto_mechanisms = {
        'NoVeto': VetoMechanism(),  # Base class that doesn't veto anything
        'RandomVeto': RandomVetoMechanism(veto_probability=0.2),
        'FixedRules': FixedRulesVetoMechanism(),
        'QValueThreshold': QValueThresholdVeto(threshold_factor=0.6),
        'Historical': HistoricalPerformanceVeto(history_size=50, poor_threshold=0.0),
        'ThresholdVeto': ThresholdVetoMechanism(threshold=0.7),
        'UncertaintyVeto': UncertaintyVetoMechanism(uncertainty_estimator, uncertainty_threshold=0.5)
    }
    
    # Run benchmarks for all combinations (except incompatible ones)
    results = {}
    
    # Define which combinations to benchmark
    # Not all agent-veto combinations make sense (e.g., Random agent with UncertaintyVeto)
    benchmark_combinations = [
        ('Random', 'NoVeto'),
        ('Random', 'RandomVeto'),
        ('Random', 'FixedRules'),
        ('RuleBased', 'NoVeto'),
        ('RuleBased', 'FixedRules'),
        ('RuleBased', 'QValueThreshold'),
        ('SimpleQLearning', 'NoVeto'),
        ('SimpleQLearning', 'QValueThreshold'),
        ('SimpleQLearning', 'Historical'),
        ('DQN', 'NoVeto'),
        ('DQN', 'QValueThreshold'),
        ('DQN', 'ThresholdVeto'),
        ('DQN', 'UncertaintyVeto'),
        ('DQN', 'Historical')
    ]
    
    for agent_name, veto_name in benchmark_combinations:
        agent = agents[agent_name]
        veto = veto_mechanisms[veto_name]
        
        # Run benchmark
        key = f"{agent_name}_{veto_name}"
        results[key] = run_benchmark(agent, veto, env, args.episodes, args.steps)
        
        # Small delay to let system resources recover
        time.sleep(1)
    
    # Save results
    import json
    results_file = os.path.join(args.save_dir, 'benchmark_results.json')
    with open(results_file, 'w') as f:
        # Convert numpy values to Python types
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                serializable_results[k] = {
                    kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                    for kk, vv in v.items()
                }
            else:
                serializable_results[k] = v
        
        json.dump(serializable_results, f, indent=2)
    
    # Generate visualization
    visualize_results(results, args.save_dir)
    
    print(f"\nBenchmark complete. Results saved to {results_file}")
    print(f"Visualizations saved to {args.save_dir}")

def visualize_results(results, save_dir):
    """Generate visualizations of benchmark results"""
    # Extract metrics by agent and veto type
    metrics_by_agent = defaultdict(list)
    metrics_by_veto = defaultdict(list)
    all_combinations = []
    
    for key, data in results.items():
        agent_name, veto_name = key.split('_')
        metrics_by_agent[agent_name].append((veto_name, data['avg_reward']))
        metrics_by_veto[veto_name].append((agent_name, data['avg_reward']))
        all_combinations.append((key, data['avg_reward'], data['successful_veto_ratio']))
    
    # Sort by average reward
    all_combinations.sort(key=lambda x: x[1], reverse=True)
    
    # 1. Bar chart of average reward by agent
    plt.figure(figsize=(12, 6))
    agents = sorted(metrics_by_agent.keys())
    veto_types = sorted({veto for agent in metrics_by_agent.values() for veto, _ in agent})
    
    x = np.arange(len(agents))
    width = 0.8 / len(veto_types)
    
    for i, veto in enumerate(veto_types):
        rewards = []
        for agent in agents:
            # Find the reward for this agent-veto combination
            reward = next((r for v, r in metrics_by_agent[agent] if v == veto), 0)
            rewards.append(reward)
        
        plt.bar(x + i * width - 0.4 + width/2, rewards, width, label=veto)
    
    plt.xlabel('Agent Type')
    plt.ylabel('Average Reward')
    plt.title('Average Reward by Agent and Veto Mechanism')
    plt.xticks(x, agents)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'reward_by_agent.png'))
    plt.close()
    
    # 2. Bar chart of average reward by veto mechanism
    plt.figure(figsize=(12, 6))
    vetos = sorted(metrics_by_veto.keys())
    agent_types = sorted({agent for veto in metrics_by_veto.values() for agent, _ in veto})
    
    x = np.arange(len(vetos))
    width = 0.8 / len(agent_types)
    
    for i, agent in enumerate(agent_types):
        rewards = []
        for veto in vetos:
            # Find the reward for this agent-veto combination
            reward = next((r for a, r in metrics_by_veto[veto] if a == agent), 0)
            rewards.append(reward)
        
        plt.bar(x + i * width - 0.4 + width/2, rewards, width, label=agent)
    
    plt.xlabel('Veto Mechanism')
    plt.ylabel('Average Reward')
    plt.title('Average Reward by Veto Mechanism and Agent')
    plt.xticks(x, vetos, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reward_by_veto.png'))
    plt.close()
    
    # 3. Scatter plot of reward vs. successful veto ratio
    plt.figure(figsize=(10, 6))
    
    combinations = [c[0] for c in all_combinations]
    rewards = [c[1] for c in all_combinations]
    veto_ratios = [c[2] for c in all_combinations]
    
    plt.scatter(veto_ratios, rewards, s=100, alpha=0.7)
    
    # Add labels to points
    for i, combo in enumerate(combinations):
        plt.annotate(combo, (veto_ratios[i], rewards[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Successful Veto Ratio')
    plt.ylabel('Average Reward')
    plt.title('Reward vs. Successful Veto Ratio')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, 'reward_vs_veto_ratio.png'))
    plt.close()
    
    # 4. Summary table as image
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    
    # Create a formatted table of results
    table_data = [['Agent + Veto', 'Avg Reward', 'Successful Veto Ratio', 'Total Vetos']]
    for key, data in sorted(results.items(), key=lambda x: x[1]['avg_reward'], reverse=True):
        table_data.append([
            key,
            f"{data['avg_reward']:.2f}",
            f"{data['successful_veto_ratio']:.2f}",
            f"{data['total_vetos']}"
        ])
    
    table = plt.table(
        cellText=table_data,
        colLabels=None,
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.2, 0.2, 0.2]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title('Benchmark Results Summary', fontsize=16, y=0.9)
    plt.savefig(os.path.join(save_dir, 'benchmark_summary.png'))
    plt.close()