#!/usr/bin/env python3
"""
Script to run the improved experiment design.
This demonstrates how to use the ImprovedExperiment class.
"""

import argparse
import sys
import os
from pathlib import Path
import itertools

# Ensure the src package is in the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.experiment.improved_experiment import ImprovedExperiment

def main():
    parser = argparse.ArgumentParser(description='Run improved experiment design')
    parser.add_argument('--output_dir', type=str, default='data/experiment_results', 
                      help='Directory to store experiment results')
    parser.add_argument('--participants', type=int, default=30, 
                      help='Number of simulated participants')
    parser.add_argument('--seed', type=int, default=None, 
                      help='Random seed for reproducibility')
    parser.add_argument('--analyze', action='store_true', 
                      help='Run analysis after experiment completion')
    parser.add_argument('--analyze_only', type=str, default=None, 
                      help='Analyze results from a previous experiment ID without running a new experiment')
    
    args = parser.parse_args()
    
    # Check if we just want to analyze existing results
    if args.analyze_only:
        # Create experiment object with the given ID
        experiment = ImprovedExperiment(
            experiment_id=args.analyze_only,
            output_dir=args.output_dir,
            seed=args.seed
        )
        
        # Run analysis
        analysis_results = experiment.analyze_results()
        
        # Print key results
        if 'hypothesis_results' in analysis_results:
            print("\nHypothesis Test Results:")
            for result in analysis_results['hypothesis_results']:
                status = result['result']
                if status == "supported":
                    symbol = "✓"
                elif status == "contradicted":
                    symbol = "✗"
                else:
                    symbol = "?"
                
                print(f"{symbol} Hypothesis {result['hypothesis_id']}: {status.capitalize()}")
                if result['p_value'] is not None:
                    print(f"   p-value: {result['p_value']:.4f}, Effect size: {result['effect_size']:.4f}")
                    
        print(f"\nDetailed analysis saved to {args.output_dir}/{args.analyze_only}/analysis/")
        return
    
    # Create experiment
    experiment = ImprovedExperiment(
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # Add formal hypotheses
    experiment.add_hypothesis(
        "Veto mechanisms using uncertainty estimation lead to higher rewards than threshold-based veto mechanisms",
        null_hypothesis="There is no difference in rewards between uncertainty-based and threshold-based veto mechanisms",
        type="superiority",
        min_effect_size=0.3
    )
    
    experiment.add_hypothesis(
        "Uncertainty-based veto mechanisms result in more efficient veto decisions (higher ratio of successful vetos)",
        type="superiority",
        min_effect_size=0.3
    )
    
    # Set up conditions with proper control
    experiment.setup_conditions(
        ["threshold", "uncertainty"],
        balanced=True,
        include_control=True
    )
    
    # Create standardized test scenarios
    experiment.setup_scenarios(
        num_scenarios=5,
        difficulty_range=(0.2, 0.8)
    )
    
    # Set up participants with counterbalancing
    experiment.setup_participants(
        num_participants=args.participants,
        virtual=True,  # Simulate participants
        balanced=True  # Use balanced design
    )
    
    # Print experiment details
    print(f"Running improved experiment with ID: {experiment.experiment_id}")
    print(f"Testing {len(experiment.config['hypotheses'])} formal hypotheses")
    print(f"Using {len(experiment.conditions)} conditions with {args.participants} participants")
    
    # Check if we have enough statistical power
    power_analysis = experiment.config.get("power_analysis", {})
    if power_analysis:
        estimated_power = power_analysis.get("estimated_power", 0)
        recommended_sample = power_analysis.get("recommended_sample_size", 0)
        
        print(f"Power analysis: Estimated power with {args.participants} participants: {estimated_power:.2f}")
        if estimated_power < 0.8:
            print(f"Warning: Experiment may be underpowered. Recommended sample size: {recommended_sample}")
    
    # Run the experiment
    output_dir = experiment.run_experiment()
    
    # Run analysis if requested
    if args.analyze:
        analysis_results = experiment.analyze_results()
        
        # Print key results
        if 'hypothesis_results' in analysis_results:
            print("\nHypothesis Test Results:")
            for result in analysis_results['hypothesis_results']:
                status = result['result']
                if status == "supported":
                    symbol = "✓"
                elif status == "contradicted":
                    symbol = "✗"
                else:
                    symbol = "?"
                
                print(f"{symbol} Hypothesis {result['hypothesis_id']}: {status.capitalize()}")
                if result['p_value'] is not None:
                    print(f"   p-value: {result['p_value']:.4f}, Effect size: {result['effect_size']:.4f}")
    
    print(f"\nExperiment complete. Results saved to {output_dir}")
    print("To analyze results later, run:")
    print(f"  python scripts/run_improved_experiment.py --analyze_only {experiment.experiment_id}")

if __name__ == "__main__":
    main()