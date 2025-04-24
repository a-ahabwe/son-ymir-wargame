#!/usr/bin/env python3
"""
Script to validate uncertainty estimation in the RL model.
Run this script to test if uncertainty estimates are meaningful.
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure the src package is in the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.game.environment import GameEnvironment
from src.ai.models import DuelingDQN
from src.ai.agent import RLAgent
from src.ai.uncertainty import UncertaintyEstimator
from src.ai.uncertainty_validation import UncertaintyValidator

def main():
    parser = argparse.ArgumentParser(description='Validate uncertainty estimates')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=20, help='Number of validation episodes')
    parser.add_argument('--save_dir', type=str, default='data/uncertainty_validation', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create environment
    env = GameEnvironment(grid_size=50)
    
    # Create agent
    agent = RLAgent(env.state_size, env.action_space_n)
    
    # Load model if path provided
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        agent.load(args.model_path)
    else:
        print("Using untrained model - results will reflect random initialization")
    
    # Create uncertainty estimator
    uncertainty_estimator = UncertaintyEstimator(agent.model, mc_dropout_samples=10)
    
    # Create validator
    validator = UncertaintyValidator(agent.model, uncertainty_estimator, env, save_dir=args.save_dir)
    
    # Run validations
    print(f"Running uncertainty validation with {args.episodes} episodes...")
    results = validator.run_all_validations()
    
    # Output core metrics
    uncertainty_quality_score = (
        results['error_correlation']['pearson_correlation'] + 
        min(1, max(0, results['decision_quality']['reward_difference']))
    ) / 2 * 100
    
    print("\n==== UNCERTAINTY VALIDATION RESULTS ====")
    print(f"Overall uncertainty quality score: {uncertainty_quality_score:.1f}%")
    
    # Detailed assessment
    print("\nDetailed Assessment:")
    
    # Error correlation assessment
    pearson = results['error_correlation']['pearson_correlation']
    if pearson > 0.5:
        print("✓ EXCELLENT: Strong correlation between uncertainty and prediction error")
    elif pearson > 0.3:
        print("✓ GOOD: Moderate correlation between uncertainty and prediction error")
    elif pearson > 0.1:
        print("⚠ FAIR: Weak correlation between uncertainty and prediction error")
    else:
        print("✗ POOR: No meaningful correlation between uncertainty and prediction error")
    
    # Decision quality assessment
    reward_diff = results['decision_quality']['reward_difference']
    p_value = results['decision_quality']['p_value']
    
    if reward_diff > 0 and p_value < 0.05:
        print("✓ EXCELLENT: Low uncertainty actions yield significantly better rewards")
    elif reward_diff > 0:
        print("✓ GOOD: Low uncertainty actions yield better rewards (not statistically significant)")
    elif reward_diff < 0 and p_value < 0.05:
        print("✗ VERY POOR: High uncertainty actions yield significantly better rewards than low uncertainty ones")
    else:
        print("⚠ POOR: No clear relationship between uncertainty and reward outcomes")
    
    # Recommendations
    print("\nRecommendations:")
    if uncertainty_quality_score < 30:
        print("- Consider using a different uncertainty estimation method")
        print("- Try increasing dropout probability to get more diverse samples")
        print("- Explore ensemble methods instead of or in addition to MC dropout")
    elif uncertainty_quality_score < 60:
        print("- Fine-tune the MC dropout rate for better calibration")
        print("- Add more training data in areas with high uncertainty")
        print("- Consider adjusting the uncertainty threshold for veto decisions")
    else:
        print("- Current uncertainty estimation appears effective")
        print("- Fine-tune veto thresholds based on validation results")
    
    print(f"\nDetailed validation results and visualizations saved to {args.save_dir}")

if __name__ == "__main__":
    main()