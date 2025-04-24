#!/usr/bin/env python3
"""
Integration script to run the improved veto system with all new features.
Demonstrates the complete workflow from training to evaluation.
"""

import os
import sys
import argparse
import time
import random
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

# Ensure the src package is in the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import improved components
from src.game.environment import GameEnvironment
from src.game.state import GameState
from src.game.state_adapter import StateAdapter
from src.ai.agent import RLAgent
from src.ai.improved_uncertainty import ImprovedUncertaintyEstimator
from src.veto.improved_risk_assessment import ImprovedRiskAssessor
from src.veto.veto_mechanism import VetoMechanism, VetoDecision
from src.veto.ground_truth import GroundTruthOracle
from src.veto.veto_validator import VetoValidator
from src.veto.veto_dataset import VetoDataset, VetoDatasetGenerator
from src.veto.selection_bias_mitigation import CounterfactualEstimator, OffPolicyEvaluator
from tests.test_framework import run_standard_veto_tests

class ImprovedVetoMechanism(VetoMechanism):
    """
    Improved veto mechanism that integrates all enhanced features.
    Combines structured state representation, ground truth validation,
    improved uncertainty estimation, and selection bias mitigation.
    """
    def __init__(self, uncertainty_estimator=None, risk_assessor=None, 
                off_policy_evaluator=None, uncertainty_threshold=0.5, timeout=10):
        """
        Initialize improved veto mechanism
        
        Args:
            uncertainty_estimator: Estimator for model uncertainty
            risk_assessor: Risk assessor for analyzing action risk
            off_policy_evaluator: Evaluator for handling selection bias
            uncertainty_threshold: Threshold for uncertainty-based vetoes
            timeout: Timeout for veto decisions in seconds
        """
        super().__init__(timeout)
        
        # Components
        self.uncertainty_estimator = uncertainty_estimator
        self.risk_assessor = risk_assessor or ImprovedRiskAssessor()
        self.off_policy_evaluator = off_policy_evaluator
        
        # Settings
        self.uncertainty_threshold = uncertainty_threshold
        self.veto_cooldown = 0  # Steps since last veto
        self.min_cooldown = 2   # Minimum steps between vetoes
        self.consecutive_veto_count = 0
        self.max_consecutive_vetoes = 3
        
        # Performance metrics
        self.metrics = {
            'veto_count': 0,
            'successful_veto_count': 0,
            'false_veto_count': 0,
            'missed_veto_count': 0,
            'decision_count': 0
        }
        
        # Recent veto decisions for analysis
        self.recent_decisions = []
        self.max_decisions = 100
    
    def assess_action(self, state, action, q_values=None, uncertainty=None):
        """
        Assess whether an action should be vetoed
        
        Args:
            state: Current state
            action: Action to assess
            q_values: Optional Q-values for all actions
            uncertainty: Optional pre-computed uncertainty value
            
        Returns:
            VetoDecision object with assessment results
        """
        # Convert state to GameState if needed
        if not isinstance(state, GameState):
            try:
                state = GameState(raw_state=state)
            except:
                # If conversion fails, proceed with raw state
                pass
                
        # Track metrics
        self.metrics['decision_count'] += 1
        
        # 1. Check risk assessment from risk assessor
        is_risky, risk_reason = self.risk_assessor.is_high_risk(state, action, q_values)
        
        # 2. Get uncertainty if estimator is available
        if uncertainty is None and self.uncertainty_estimator is not None:
            uncertainty = self.uncertainty_estimator.get_action_uncertainty(state, action)
        else:
            uncertainty = 0.0
            
        # 3. Determine if veto is needed
        should_veto = False
        veto_reason = ""
        
        # Veto for high risk
        if is_risky:
            should_veto = True
            veto_reason = risk_reason
        
        # Veto for high uncertainty
        if uncertainty > self.uncertainty_threshold:
            should_veto = True
            if veto_reason:
                veto_reason += "; "
            veto_reason += f"High uncertainty ({uncertainty:.2f})"
            
        # 4. Apply veto cooldown and consecutive veto limit
        if should_veto:
            # Check cooldown
            if self.veto_cooldown < self.min_cooldown:
                should_veto = False
                veto_reason = f"Veto suppressed (cooldown: {self.veto_cooldown}/{self.min_cooldown})"
            
            # Check consecutive veto count
            if should_veto:
                self.consecutive_veto_count += 1
                if self.consecutive_veto_count > self.max_consecutive_vetoes:
                    should_veto = False
                    veto_reason = "Veto suppressed (too many consecutive vetoes)"
                    self.consecutive_veto_count = 0
        else:
            # Reset consecutive veto count
            self.consecutive_veto_count = 0
            
        # Update cooldown
        if should_veto:
            self.veto_cooldown = 0
        else:
            self.veto_cooldown += 1
            
        # Update metrics
        if should_veto:
            self.metrics['veto_count'] += 1
            
        # Create decision
        decision = VetoDecision(
            original_action=action,
            vetoed=should_veto,
            reason=veto_reason,
            risk_reason=risk_reason,
            uncertainty=uncertainty,
            threshold=self.uncertainty_threshold,
            q_values=q_values
        )
        
        # Record decision for analysis
        self._record_decision({
            'state': state.raw if isinstance(state, GameState) else state,
            'action': action,
            'q_values': q_values.tolist() if q_values is not None and hasattr(q_values, 'tolist') else q_values,
            'uncertainty': uncertainty,
            'is_risky': is_risky,
            'risk_reason': risk_reason,
            'vetoed': should_veto,
            'veto_reason': veto_reason,
            'timestamp': time.time()
        })
        
        return decision
        
    def record_veto_decision(self, state, action, vetoed, alternative=None, outcome=None):
        """
        Record a veto decision and its outcome for learning
        
        Args:
            state: State where decision was made
            action: Original action
            vetoed: Whether action was vetoed
            alternative: Alternative action if vetoed
            outcome: Outcome of the decision
        """
        super().record_veto_decision(state, action, vetoed, alternative, outcome)
        
        # Process outcome for metrics
        if vetoed and outcome is not None:
            # Extract reward
            if isinstance(outcome, tuple):
                reward = outcome[0]
            else:
                reward = outcome
                
            # Record whether veto was successful (positive outcome)
            if reward > 0:
                self.metrics['successful_veto_count'] += 1
            else:
                self.metrics['false_veto_count'] += 1
                
        # Process with off-policy evaluator if available
        if self.off_policy_evaluator is not None:
            learning_data = self.off_policy_evaluator.process_veto_decision(
                state, action, vetoed, alternative, outcome
            )
            
            # Return learning data for potential further use
            return learning_data
    
    def _record_decision(self, decision_data):
        """Record decision for analysis"""
        # Add to recent decisions
        self.recent_decisions.append(decision_data)
        
        # Trim if too many
        if len(self.recent_decisions) > self.max_decisions:
            self.recent_decisions = self.recent_decisions[-self.max_decisions:]
    
    def get_metrics(self):
        """Get mechanism performance metrics"""
        # Calculate derived metrics
        metrics = self.metrics.copy()
        
        # Calculate veto success rate if any vetoes
        if metrics['veto_count'] > 0:
            metrics['veto_success_rate'] = metrics['successful_veto_count'] / metrics['veto_count']
        else:
            metrics['veto_success_rate'] = 0.0
            
        # Calculate decision statistics
        metrics['veto_rate'] = metrics['veto_count'] / max(1, metrics['decision_count'])
        
        return metrics
    
    def analyze_decisions(self):
        """Analyze recent decisions for insights"""
        if not self.recent_decisions:
            return None
            
        # Analyze veto distribution
        veto_count = sum(1 for d in self.recent_decisions if d['vetoed'])
        non_veto_count = len(self.recent_decisions) - veto_count
        
        # Analyze decision factors
        risk_factor_count = sum(1 for d in self.recent_decisions 
                              if d['vetoed'] and 'risk' in d['veto_reason'].lower())
        uncertainty_factor_count = sum(1 for d in self.recent_decisions 
                                     if d['vetoed'] and 'uncertainty' in d['veto_reason'].lower())
        
        # Q-value analysis if available
        q_value_data = [d for d in self.recent_decisions if d['q_values'] is not None]
        if q_value_data:
            avg_q_vetoed = np.mean([
                d['q_values'][d['action']] for d in q_value_data if d['vetoed']
            ]) if any(d['vetoed'] for d in q_value_data) else None
            
            avg_q_non_vetoed = np.mean([
                d['q_values'][d['action']] for d in q_value_data if not d['vetoed']
            ]) if any(not d['vetoed'] for d in q_value_data) else None
        else:
            avg_q_vetoed = None
            avg_q_non_vetoed = None
        
        # Uncertainty analysis
        if any('uncertainty' in d for d in self.recent_decisions):
            avg_uncertainty_vetoed = np.mean([
                d['uncertainty'] for d in self.recent_decisions 
                if d['vetoed'] and 'uncertainty' in d
            ]) if any(d['vetoed'] and 'uncertainty' in d for d in self.recent_decisions) else None
            
            avg_uncertainty_non_vetoed = np.mean([
                d['uncertainty'] for d in self.recent_decisions 
                if not d['vetoed'] and 'uncertainty' in d
            ]) if any(not d['vetoed'] and 'uncertainty' in d for d in self.recent_decisions) else None
        else:
            avg_uncertainty_vetoed = None
            avg_uncertainty_non_vetoed = None
        
        return {
            'veto_count': veto_count,
            'non_veto_count': non_veto_count,
            'veto_rate': veto_count / len(self.recent_decisions),
            'risk_factor_count': risk_factor_count,
            'uncertainty_factor_count': uncertainty_factor_count,
            'avg_q_vetoed': avg_q_vetoed,
            'avg_q_non_vetoed': avg_q_non_vetoed,
            'avg_uncertainty_vetoed': avg_uncertainty_vetoed,
            'avg_uncertainty_non_vetoed': avg_uncertainty_non_vetoed
        }


def create_improved_components(model_path=None, num_ensemble=3, mc_samples=10):
    """
    Create improved veto system components
    
    Args:
        model_path: Path to pre-trained model
        num_ensemble: Number of models for ensemble uncertainty
        mc_samples: Number of samples for MC dropout
    
    Returns:
        Tuple of (agent, uncertainty_estimator, risk_assessor, off_policy_evaluator)
    """
    # Create environment for state dimensionality
    env = GameEnvironment(grid_size=50)
    
    # Create agent
    agent = RLAgent(env.state_size, env.action_space_n)
    
    # Load model if provided
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        agent.load(model_path)
    
    # Create improved uncertainty estimator
    uncertainty_estimator = ImprovedUncertaintyEstimator(
        agent.model, 
        num_models=num_ensemble,
        mc_dropout_samples=mc_samples
    )
    
    # Create improved risk assessor
    risk_assessor = ImprovedRiskAssessor()
    
    # Create counterfactual estimator
    counterfactual_estimator = CounterfactualEstimator(
        environment_class=GameEnvironment
    )
    
    # Create off-policy evaluator
    off_policy_evaluator = OffPolicyEvaluator(
        counterfactual_estimator=counterfactual_estimator
    )
    
    return agent, uncertainty_estimator, risk_assessor, off_policy_evaluator


def run_benchmark(veto_mechanism, agent, env, num_episodes=20, max_steps=200):
    """
    Run benchmark to evaluate veto mechanism performance
    
    Args:
        veto_mechanism: Veto mechanism to evaluate
        agent: Agent to use for action selection
        env: Environment to run in
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary with benchmark results
    """
    # Track metrics
    episode_rewards = []
    episode_lengths = []
    veto_counts = []
    successful_vetos = []
    
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
            decision = veto_mechanism.assess_action(state, action, q_values)
            
            if decision.vetoed:
                episode_veto_count += 1
                # Use agent's safe action selection
                if hasattr(agent, 'select_safe_action'):
                    alternative_action = agent.select_safe_action(state, action)
                else:
                    # Simple rule: choose a different action
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
                    outcome=reward
                )
            else:
                # Execute original action
                next_state, reward, done, _ = env.step(action)
                
                # Record non-veto
                veto_mechanism.record_veto_decision(
                    state=state,
                    action=action,
                    vetoed=False,
                    alternative=None,
                    outcome=reward
                )
            
            # Store transition for learning
            agent.store_transition(state, action, reward, next_state, done)
            
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
        'successful_veto_ratio': successful_veto_ratio,
        'mechanism_metrics': veto_mechanism.get_metrics()
    }


def compare_mechanisms(agent, mechanisms, env, num_episodes=10, save_dir=None):
    """
    Compare multiple veto mechanisms
    
    Args:
        agent: Agent to use for action selection
        mechanisms: Dictionary mapping names to veto mechanisms
        env: Environment to run in
        num_episodes: Number of episodes to run per mechanism
        save_dir: Directory to save results
        
    Returns:
        Dictionary with comparison results
    """
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    # Run benchmark for each mechanism
    results = {}
    
    for name, mechanism in mechanisms.items():
        print(f"\nBenchmarking {name}...")
        
        # Run benchmark
        results[name] = run_benchmark(mechanism, agent, env, num_episodes)
        
        # Save individual results
        if save_dir:
            mech_path = os.path.join(save_dir, f"{name}_results.json")
            import json
            with open(mech_path, 'w') as f:
                json.dump(
                    {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                     for k, v in results[name].items()},
                    f, indent=2, default=str
                )
    
    # Create comparison visualizations
    if save_dir:
        _create_comparison_visualizations(results, save_dir)
    
    return results


def _create_comparison_visualizations(results, save_dir):
    """Create visualizations comparing veto mechanisms"""
    # 1. Reward comparison
    plt.figure(figsize=(10, 6))
    
    names = list(results.keys())
    avg_rewards = [r['avg_reward'] for r in results.values()]
    
    plt.bar(names, avg_rewards)
    plt.title('Average Reward by Veto Mechanism')
    plt.ylabel('Average Reward')
    plt.savefig(os.path.join(save_dir, 'reward_comparison.png'))
    plt.close()
    
    # 2. Veto rate vs. successful veto ratio
    plt.figure(figsize=(10, 6))
    
    veto_rates = [r['total_vetos'] / (r['avg_length'] * len(r['episode_rewards'])) 
                for r in results.values()]
    success_ratios = [r['successful_veto_ratio'] for r in results.values()]
    
    plt.scatter(veto_rates, success_ratios, s=100)
    
    # Add labels
    for i, name in enumerate(names):
        plt.annotate(name, (veto_rates[i], success_ratios[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Veto Rate')
    plt.ylabel('Successful Veto Ratio')
    plt.title('Veto Rate vs. Success Ratio')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, 'veto_rate_vs_success.png'))
    plt.close()
    
    # 3. Reward over time
    plt.figure(figsize=(12, 6))
    
    # Get max episode count
    max_episodes = max(len(r['episode_rewards']) for r in results.values())
    episodes = list(range(1, max_episodes + 1))
    
    for name, data in results.items():
        rewards = data['episode_rewards']
        plt.plot(episodes[:len(rewards)], rewards, label=name)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward by Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'reward_over_time.png'))
    plt.close()


def generate_validation_dataset(env, agent, oracle, size=500):
    """
    Generate a validation dataset with ground truth labels
    
    Args:
        env: Environment to use
        agent: Agent for selecting actions
        oracle: Ground truth oracle
        size: Number of examples to generate
        
    Returns:
        Generated VetoDataset
    """
    print(f"Generating validation dataset with {size} examples...")
    
    # Create dataset
    dataset = VetoDataset("validation_dataset")
    
    # Generate examples
    examples_collected = 0
    
    while examples_collected < size:
        # Reset environment
        state = env.reset()
        done = False
        
        # Run an episode
        while not done and examples_collected < size:
            # Select action
            action, q_values = agent.select_action(state)
            
            # Get ground truth
            should_veto, confidence, explanation = oracle.should_veto(state, action, q_values)
            
            # Add to dataset
            dataset.add_example(state, action, should_veto, q_values, confidence, explanation)
            examples_collected += 1
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            state = next_state
            
            # Print progress
            if examples_collected % 50 == 0:
                print(f"  Generated {examples_collected}/{size} examples")
    
    # Balance dataset
    dataset.balance_dataset()
    
    print(f"Dataset generation complete: {len(dataset.data)} examples "
         f"({dataset.metadata['statistics']['veto_count']} veto, "
         f"{dataset.metadata['statistics']['noveto_count']} no-veto)")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Run improved veto system')
    parser.add_argument('--model_path', type=str, default=None, 
                      help='Path to pre-trained model')
    parser.add_argument('--mode', type=str, choices=['benchmark', 'validate', 'test'], 
                      default='benchmark', help='Operation mode')
    parser.add_argument('--episodes', type=int, default=20, 
                      help='Number of episodes to run')
    parser.add_argument('--save_dir', type=str, default='data/improved_veto', 
                      help='Directory to save results')
    parser.add_argument('--generate_dataset', action='store_true', 
                      help='Generate validation dataset')
    parser.add_argument('--dataset_size', type=int, default=500, 
                      help='Size of validation dataset to generate')
    parser.add_argument('--verbose', action='store_true', 
                      help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create components
    agent, uncertainty_estimator, risk_assessor, off_policy_evaluator = create_improved_components(
        model_path=args.model_path
    )
    
    # Create environment
    env = GameEnvironment(grid_size=50)
    
    # Create improved veto mechanism
    improved_veto = ImprovedVetoMechanism(
        uncertainty_estimator=uncertainty_estimator,
        risk_assessor=risk_assessor,
        off_policy_evaluator=off_policy_evaluator,
        uncertainty_threshold=0.5
    )
    
    # Create baseline mechanisms for comparison
    from src.veto.veto_mechanism import ThresholdVetoMechanism, UncertaintyVetoMechanism
    
    mechanisms = {
        'Improved': improved_veto,
        'Threshold': ThresholdVetoMechanism(threshold=0.7),
        'Uncertainty': UncertaintyVetoMechanism(
            uncertainty_estimator=uncertainty_estimator,
            uncertainty_threshold=0.5
        )
    }
    
    # Run in appropriate mode
    if args.mode == 'benchmark':
        # Run benchmark comparison
        print("Running benchmark comparison...")
        results = compare_mechanisms(
            agent, 
            mechanisms, 
            env, 
            num_episodes=args.episodes,
            save_dir=args.save_dir
        )
        
        # Print summary
        print("\nBenchmark Summary:")
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  Average Reward: {result['avg_reward']:.2f}")
            print(f"  Successful Veto Ratio: {result['successful_veto_ratio']:.2f}")
            print(f"  Total Vetos: {result['total_vetos']}")
        
    elif args.mode == 'validate':
        # Create ground truth oracle
        oracle = GroundTruthOracle(GameEnvironment)
        
        # Generate or load validation dataset
        if args.generate_dataset:
            dataset = generate_validation_dataset(
                env, agent, oracle, size=args.dataset_size
            )
            
            # Save dataset
            dataset_dir = os.path.join(args.save_dir, 'datasets')
            os.makedirs(dataset_dir, exist_ok=True)
            dataset.save(dataset_dir)
            print(f"Dataset saved to {dataset_dir}")
        else:
            # Try to load existing dataset
            dataset_dir = os.path.join(args.save_dir, 'datasets')
            if os.path.exists(os.path.join(dataset_dir, 'validation_dataset_metadata.json')):
                dataset = VetoDataset.load(dataset_dir, 'validation_dataset')
                print(f"Loaded dataset with {len(dataset.data)} examples")
            else:
                print("No dataset found, generating new one...")
                dataset = generate_validation_dataset(
                    env, agent, oracle, size=args.dataset_size
                )
                # Save dataset
                os.makedirs(dataset_dir, exist_ok=True)
                dataset.save(dataset_dir)
                print(f"Dataset saved to {dataset_dir}")
        
        # Create validator
        validator = VetoValidator(GameEnvironment, oracle)
        
        # Validate each mechanism
        validation_results = {}
        for name, mechanism in mechanisms.items():
            print(f"\nValidating {name}...")
            
            # Test on a sample from the dataset
            test_sample = dataset.get_batch(100, balanced=True)
            
            correct_count = 0
            total_count = 0
            
            for example in test_sample:
                state = example['state_array']
                action = example['action']
                q_values = example['q_values']
                ground_truth = example['should_veto']
                
                # Get mechanism decision
                decision = mechanism.assess_action(state, action, q_values)
                
                # Check correctness
                correct = decision.vetoed == ground_truth
                if correct:
                    correct_count += 1
                total_count += 1
            
            # Calculate accuracy
            accuracy = correct_count / total_count if total_count > 0 else 0
            
            validation_results[name] = {
                'accuracy': accuracy,
                'correct_count': correct_count,
                'total_count': total_count
            }
            
            print(f"  Accuracy: {accuracy:.2f} ({correct_count}/{total_count})")
        
        # Save validation results
        import json
        with open(os.path.join(args.save_dir, 'validation_results.json'), 'w') as f:
            json.dump(validation_results, f, indent=2)
        
    elif args.mode == 'test':
        # Run standardized tests
        print("Running standardized tests...")
        run_standard_veto_tests(mechanisms, args.save_dir)
    
    print("\nDone!")

if __name__ == "__main__":
    main()