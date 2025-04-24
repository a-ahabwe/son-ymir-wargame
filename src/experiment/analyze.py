#!/usr/bin/env python3
"""
Analysis script for Veto Game experiments.
Analyzes collected data and generates reports.
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Ensure the src package is in the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.experiment.analysis import ExperimentAnalysis

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results for Veto Game')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory to analyze')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist")
        return
    
    print(f"Analyzing data in {args.data_dir}")
    
    # List files in directory
    json_files = [f for f in os.listdir(args.data_dir) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files: {', '.join(json_files)}")
    
    # Create analyzer
    analyzer = ExperimentAnalysis(args.data_dir)
    
    # Generate report
    report = analyzer.generate_report()
    
    print("Analysis complete!")
    print(f"Report saved to {args.data_dir}/analysis/analysis_report.json")
    print(f"Plots saved to {args.data_dir}/analysis/")
    
    # Print summary findings
    if hasattr(analyzer, 'sessions_df') and analyzer.sessions_df is not None and not analyzer.sessions_df.empty:
        conditions = analyzer.sessions_df['condition'].unique()
        print("\nSummary Results:")
        print(f"Total participants: {len(analyzer.sessions_df['participant_id'].unique())}")
        print(f"Conditions: {', '.join(conditions)}")
        
        # Performance by condition
        performance = analyzer.analyze_performance_by_condition()
        if 'condition_metrics' in performance:
            print("\nPerformance by Condition:")
            print(performance['condition_metrics'])
            
        # Veto outcomes
        veto_outcomes = analyzer.analyze_veto_outcomes()
        if 'mean_rewards' in veto_outcomes:
            print("\nVeto Decision Outcomes:")
            print(veto_outcomes['mean_rewards'])
            
            if 'stats_result' in veto_outcomes and veto_outcomes['stats_result']:
                p_value = veto_outcomes['stats_result']['p_value']
                print(f"Significance: p = {p_value:.4f}")
    else:
        print("\nWarning: No session data was successfully loaded.")
        print("Data structure issues:")
        
        # Check specific files to diagnose
        for filename in json_files:
            if filename.startswith('p') and filename.endswith('.json'):
                file_path = os.path.join(args.data_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        print(f"  {filename}: Contains {len(data.get('actions', []))} actions, {len(data.get('veto_decisions', []))} veto decisions")
                        if args.verbose:
                            print(f"    Keys: {list(data.keys())}")
                except Exception as e:
                    print(f"  {filename}: Error - {str(e)}")
                    
        print("\nPlease check that data files include actions with 'reward' field and/or veto_decisions with outcome data.")
        print("If running in headless mode, make sure the experiment is configured to log all necessary data.")

if __name__ == "__main__":
    main()