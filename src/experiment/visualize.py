#!/usr/bin/env python3
"""
Visualization script for Veto Game experiments.
Generates additional visualizations for experiment data.
"""

import argparse
import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Ensure the src package is in the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.veto.metrics import VetoMetrics
from src.experiment.analysis import ExperimentAnalysis


def visualize_reward_distribution(data_dir, output_dir=None):
    """Create histograms of reward distributions by condition"""
    # Use ExperimentAnalysis to load the data
    analyzer = ExperimentAnalysis(data_dir)
    
    if analyzer.actions_df is None or len(analyzer.actions_df) == 0:
        print("No action data found for reward distribution visualization")
        return
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = os.path.join(data_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Group data by condition
    conditions = analyzer.actions_df['condition'].unique()
    
    # Create histogram of rewards by condition
    plt.figure(figsize=(12, 6))
    
    for condition in conditions:
        condition_rewards = analyzer.actions_df[analyzer.actions_df['condition'] == condition]['reward']
        plt.hist(condition_rewards, bins=30, alpha=0.5, label=condition)
    
    plt.title('Reward Distribution by Condition')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'reward_distribution.png'))
    plt.close()
    
    # Create a box plot
    plt.figure(figsize=(10, 6))
    data = [analyzer.actions_df[analyzer.actions_df['condition'] == c]['reward'] for c in conditions]
    plt.boxplot(data, labels=conditions)
    plt.title('Reward Distribution by Condition')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(output_dir, 'reward_boxplot.png'))
    plt.close()
    
    print(f"Reward distribution visualizations saved to {output_dir}")


def visualize_veto_metrics(data_dir, output_dir=None):
    """Extract veto data and create visualizations using VetoMetrics"""
    # Use ExperimentAnalysis to load the data
    analyzer = ExperimentAnalysis(data_dir)
    
    if analyzer.vetos_df is None or len(analyzer.vetos_df) == 0:
        print("No veto data found for veto metrics visualization")
        return
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = os.path.join(data_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a VetoMetrics instance
    metrics = VetoMetrics(save_dir=output_dir)
    
    # Convert loaded data to VetoMetrics format
    for _, veto in analyzer.vetos_df.iterrows():
        # Extract data from the veto entry
        veto_requested = True  # If it's in vetos_df, it was requested
        veto_result = veto.get('vetoed', None)
        response_time = veto.get('response_time', None)
        outcome = veto.get('reward', None)
        
        # Record in metrics
        metrics.record_action(
            veto_requested=veto_requested,
            veto_result=veto_result,
            response_time=response_time,
            outcome=outcome
        )
        
    # Generate and save metrics
    metrics_path = metrics.save_metrics('veto_analysis')
    
    print(f"Veto metrics saved to {metrics_path}")
    
    # Create additional custom visualizations
    
    # Veto rate by condition
    if 'condition' in analyzer.vetos_df.columns:
        veto_by_condition = analyzer.vetos_df.groupby(['condition', 'vetoed']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(10, 6))
        veto_by_condition.plot(kind='bar', stacked=True)
        plt.title('Veto Decisions by Condition')
        plt.xlabel('Condition')
        plt.ylabel('Count')
        plt.legend(['Accepted', 'Vetoed'])
        plt.savefig(os.path.join(output_dir, 'veto_by_condition.png'))
        plt.close()
    
    print(f"Additional veto visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations for Veto Game experiments')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory to analyze')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for visualizations')
    parser.add_argument('--reward_dist', action='store_true', help='Generate reward distribution visualizations')
    parser.add_argument('--veto_metrics', action='store_true', help='Generate veto metrics visualizations')
    parser.add_argument('--all', action='store_true', help='Generate all visualizations')
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist")
        return
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "visualizations")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate requested visualizations
    if args.all or args.reward_dist:
        visualize_reward_distribution(args.data_dir, args.output_dir)
    
    if args.all or args.veto_metrics:
        visualize_veto_metrics(args.data_dir, args.output_dir)
    
    print(f"All visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main() 