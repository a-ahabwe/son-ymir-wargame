#!/usr/bin/env python3
"""
Analyze benchmark results to determine which approaches are most effective.
This script loads previously generated benchmark data and performs statistical analysis.
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

def load_benchmark_results(results_file):
    """Load benchmark results from JSON file"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Convert to DataFrame for easier analysis
    data = []
    
    for key, metrics in results.items():
        agent_name, veto_name = key.split('_')
        
        # Extract key metrics
        data.append({
            'agent': agent_name,
            'veto': veto_name,
            'avg_reward': metrics['avg_reward'],
            'avg_length': metrics['avg_length'],
            'total_vetos': metrics['total_vetos'],
            'successful_veto_ratio': metrics['successful_veto_ratio'],
            'episode_rewards': metrics['episode_rewards'],
            'veto_counts': metrics['veto_counts']
        })
    
    return pd.DataFrame(data)

def perform_statistical_tests(df):
    """Perform statistical tests to compare approaches"""
    results = {}
    
    # 1. Effect of veto mechanisms - ANOVA
    # For each agent type, compare rewards across veto mechanisms
    for agent in df['agent'].unique():
        agent_data = df[df['agent'] == agent]
        
        if len(agent_data) >= 2:  # Need at least 2 veto types to compare
            # Extract reward data into groups by veto type
            groups = [episode_rewards for veto, episode_rewards in 
                     zip(agent_data['veto'], agent_data['episode_rewards'])]
            
            # Run ANOVA if we have enough groups
            if len(groups) >= 2:
                f_stat, p_value = stats.f_oneway(*groups)
                results[f'ANOVA_{agent}_vetos'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
    
    # 2. Effect of agents - ANOVA
    # For each veto mechanism, compare rewards across agent types
    for veto in df['veto'].unique():
        veto_data = df[df['veto'] == veto]
        
        if len(veto_data) >= 2:  # Need at least 2 agent types to compare
            # Extract reward data into groups by agent type
            groups = [episode_rewards for agent, episode_rewards in 
                     zip(veto_data['agent'], veto_data['episode_rewards'])]
            
            # Run ANOVA if we have enough groups
            if len(groups) >= 2:
                f_stat, p_value = stats.f_oneway(*groups)
                results[f'ANOVA_{veto}_agents'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
    
    # 3. Compare our approach vs baselines - t-tests
    # Find our main approach (DQN + UncertaintyVeto)
    if 'DQN' in df['agent'].values and 'UncertaintyVeto' in df['veto'].values:
        main_approach = df[(df['agent'] == 'DQN') & (df['veto'] == 'UncertaintyVeto')]
        
        if not main_approach.empty:
            main_rewards = main_approach['episode_rewards'].iloc[0]
            
            # Compare against each baseline
            for _, baseline in df.iterrows():
                if baseline['agent'] == 'DQN' and baseline['veto'] == 'UncertaintyVeto':
                    continue  # Skip comparing to itself
                
                baseline_rewards = baseline['episode_rewards']
                t_stat, p_value = stats.ttest_ind(main_rewards, baseline_rewards)
                
                results[f'ttest_vs_{baseline["agent"]}_{baseline["veto"]}'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'mean_difference': np.mean(main_rewards) - np.mean(baseline_rewards)
                }
    
    return results

def analyze_veto_effectiveness(df):
    """Analyze how effective different veto mechanisms are"""
    # Calculate correlation between veto count and reward
    correlations = {}
    
    for agent in df['agent'].unique():
        agent_data = df[df['agent'] == agent]
        
        # Skip if we have fewer than 3 veto types
        if len(agent_data) < 3:
            continue
            
        # Calculate correlation
        corr, p_value = stats.pearsonr(agent_data['total_vetos'], agent_data['avg_reward'])
        
        correlations[agent] = {
            'correlation': corr,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # Calculate veto efficiency (reward gained per veto)
    df_with_baseline = df.copy()
    
    # Add veto efficiency metric
    for agent in df['agent'].unique():
        # Find no-veto baseline for this agent
        baseline = df[(df['agent'] == agent) & (df['veto'] == 'NoVeto')]
        
        if len(baseline) == 0:
            continue
            
        baseline_reward = baseline['avg_reward'].iloc[0]
        
        # Calculate efficiency for each veto mechanism
        for idx, row in df[(df['agent'] == agent) & (df['veto'] != 'NoVeto')].iterrows():
            if row['total_vetos'] > 0:
                veto_efficiency = (row['avg_reward'] - baseline_reward) / row['total_vetos']
            else:
                veto_efficiency = 0
                
            df_with_baseline.loc[idx, 'veto_efficiency'] = veto_efficiency
    
    return {
        'correlations': correlations,
        'df_with_efficiency': df_with_baseline
    }

def visualize_analysis(df, analysis_results, save_dir):
    """Create visualizations from the analysis results"""
    # Make sure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Veto efficiency plot
    if 'veto_efficiency' in df.columns:
        plt.figure(figsize=(12, 6))
        
        # Filter out rows without efficiency data
        efficiency_data = df[df['veto_efficiency'].notna()]
        
        if not efficiency_data.empty:
            sns.barplot(data=efficiency_data, x='veto', y='veto_efficiency', hue='agent')
            plt.title('Veto Efficiency (Reward Gain per Veto)')
            plt.xlabel('Veto Mechanism')
            plt.ylabel('Efficiency')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'veto_efficiency.png'))
            plt.close()
    
    # 2. Veto count vs reward scatter plot
    plt.figure(figsize=(10, 6))
    
    for agent in df['agent'].unique():
        agent_data = df[df['agent'] == agent]
        plt.scatter(
            agent_data['total_vetos'], 
            agent_data['avg_reward'],
            label=agent,
            alpha=0.7,
            s=100
        )
    
    # Add annotations
    for _, row in df.iterrows():
        plt.annotate(
            row['veto'],
            (row['total_vetos'], row['avg_reward']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    plt.title('Veto Count vs Average Reward')
    plt.xlabel('Total Veto Count')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, 'veto_count_vs_reward.png'))
    plt.close()
    
    # 3. Statistical test results table
    stats_results = analysis_results.get('statistical_tests', {})
    
    if stats_results:
        plt.figure(figsize=(12, len(stats_results) * 0.5 + 2))
        plt.axis('off')
        
        # Create a formatted table of results
        table_data = [['Test', 'Statistic', 'p-value', 'Significant?']]
        for test_name, result in stats_results.items():
            table_data.append([
                test_name,
                f"{result.get('f_statistic') or result.get('t_statistic'):.4f}",
                f"{result['p_value']:.4f}",
                "Yes" if result['significant'] else "No"
            ])
        
        table = plt.table(
            cellText=table_data,
            colLabels=None,
            cellLoc='center',
            loc='center',
            colWidths=[0.4, 0.2, 0.2, 0.2]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        plt.title('Statistical Test Results', fontsize=16, y=0.98)
        plt.savefig(os.path.join(save_dir, 'statistical_tests.png'))
        plt.close()
    
    # 4. Correlation heatmap for agents with all veto mechanisms
    complete_agents = []
    all_vetos = set(df['veto'])
    
    for agent in df['agent'].unique():
        agent_vetos = set(df[df['agent'] == agent]['veto'])
        if agent_vetos == all_vetos:
            complete_agents.append(agent)
    
    if complete_agents:
        # Create pivot table for heatmap
        pivot_data = df[df['agent'].isin(complete_agents)].pivot(
            index='agent', columns='veto', values='avg_reward'
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Agent Performance with Different Veto Mechanisms')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_heatmap.png'))
        plt.close()

def generate_report(df, analysis_results, save_dir):
    """Generate a text report summarizing the findings"""
    report = []
    report.append("# Benchmark Analysis Report")
    report.append("\n## Overall Performance")
    
    # Top performing combinations
    top_performers = df.sort_values('avg_reward', ascending=False).head(3)
    report.append("\nTop 3 performing combinations:")
    for _, row in top_performers.iterrows():
        report.append(f"* {row['agent']} + {row['veto']}: Average Reward = {row['avg_reward']:.2f}")
    
    # Worst performing combinations
    bottom_performers = df.sort_values('avg_reward').head(3)
    report.append("\nBottom 3 performing combinations:")
    for _, row in bottom_performers.iterrows():
        report.append(f"* {row['agent']} + {row['veto']}: Average Reward = {row['avg_reward']:.2f}")
    
    # Best veto for each agent
    report.append("\n## Best Veto Mechanism for Each Agent")
    for agent in df['agent'].unique():
        agent_data = df[df['agent'] == agent]
        best_veto = agent_data.loc[agent_data['avg_reward'].idxmax()]
        report.append(f"\n### {agent}")
        report.append(f"* Best veto: {best_veto['veto']}")
        report.append(f"* Average reward: {best_veto['avg_reward']:.2f}")
        report.append(f"* Veto count: {best_veto['total_vetos']}")
        report.append(f"* Successful veto ratio: {best_veto['successful_veto_ratio']:.2f}")
    
    # Statistical test results
    if 'statistical_tests' in analysis_results:
        report.append("\n## Statistical Analysis")
        
        # ANOVA results for veto mechanisms
        anova_vetos = {k: v for k, v in analysis_results['statistical_tests'].items() 
                      if k.startswith('ANOVA_') and '_vetos' in k}
        
        if anova_vetos:
            report.append("\n### Effect of Veto Mechanisms")
            for test, result in anova_vetos.items():
                agent = test.split('_')[1]
                significance = "significant" if result['significant'] else "not significant"
                report.append(f"* For {agent}, the effect of different veto mechanisms is {significance} "
                            f"(F = {result['f_statistic']:.4f}, p = {result['p_value']:.4f})")
        
        # ANOVA results for agents
        anova_agents = {k: v for k, v in analysis_results['statistical_tests'].items() 
                       if k.startswith('ANOVA_') and '_agents' in k}
        
        if anova_agents:
            report.append("\n### Effect of Agent Types")
            for test, result in anova_agents.items():
                veto = test.split('_')[1]
                significance = "significant" if result['significant'] else "not significant"
                report.append(f"* For {veto}, the effect of different agent types is {significance} "
                            f"(F = {result['f_statistic']:.4f}, p = {result['p_value']:.4f})")
        
        # t-test results for our approach vs baselines
        ttests = {k: v for k, v in analysis_results['statistical_tests'].items() 
                 if k.startswith('ttest_vs_')}
        
        if ttests:
            report.append("\n### DQN + UncertaintyVeto vs Baselines")
            for test, result in ttests.items():
                baseline = test.replace('ttest_vs_', '')
                significance = "significantly" if result['significant'] else "not significantly"
                direction = "better than" if result['mean_difference'] > 0 else "worse than"
                report.append(f"* DQN + UncertaintyVeto performs {significance} {direction} {baseline} "
                            f"(t = {result['t_statistic']:.4f}, p = {result['p_value']:.4f}, "
                            f"mean diff = {result['mean_difference']:.2f})")
    
    # Veto effectiveness analysis
    if 'veto_effectiveness' in analysis_results:
        veto_effectiveness = analysis_results['veto_effectiveness']
        
        report.append("\n## Veto Effectiveness Analysis")
        
        # Correlation between veto count and reward
        correlations = veto_effectiveness.get('correlations', {})
        if correlations:
            report.append("\n### Correlation between Veto Count and Reward")
            for agent, corr_data in correlations.items():
                correlation = corr_data['correlation']
                significance = "significant" if corr_data['significant'] else "not significant"
                direction = "positive" if correlation > 0 else "negative"
                report.append(f"* For {agent}, there is a {significance} {direction} correlation "
                            f"(r = {correlation:.4f}, p = {corr_data['p_value']:.4f})")
        
        # Most efficient veto mechanisms
        df_with_efficiency = veto_effectiveness.get('df_with_efficiency')
        if 'veto_efficiency' in df_with_efficiency.columns:
            report.append("\n### Veto Efficiency (Reward Gain per Veto)")
            
            # Filter out rows without efficiency data and veto count = 0
            efficiency_data = df_with_efficiency[
                df_with_efficiency['veto_efficiency'].notna() & 
                (df_with_efficiency['total_vetos'] > 0)
            ]
            
            if not efficiency_data.empty:
                for agent in efficiency_data['agent'].unique():
                    agent_efficiency = efficiency_data[efficiency_data['agent'] == agent]
                    
                    # Skip if fewer than 2 veto mechanisms
                    if len(agent_efficiency) < 2:
                        continue
                        
                    # Find most efficient veto for this agent
                    best_efficiency = agent_efficiency.loc[agent_efficiency['veto_efficiency'].idxmax()]
                    
                    report.append(f"\n#### {agent}")
                    report.append(f"* Most efficient veto: {best_efficiency['veto']}")
                    report.append(f"* Efficiency: {best_efficiency['veto_efficiency']:.4f} reward per veto")
                    report.append(f"* Total vetos: {best_efficiency['total_vetos']}")
    
    # Conclusions and recommendations
    report.append("\n## Conclusions and Recommendations")
    
    # Try to determine if our approach (DQN + UncertaintyVeto) is best
    main_approach = df[(df['agent'] == 'DQN') & (df['veto'] == 'UncertaintyVeto')]
    
    if not main_approach.empty:
        # Get rank among all combinations
        df_sorted = df.sort_values('avg_reward', ascending=False)
        rank = df_sorted.index.get_loc(main_approach.index[0]) + 1
        total = len(df)
        
        if rank == 1:
            report.append("\nDQN + UncertaintyVeto is the best performing combination in our benchmarks.")
        else:
            report.append(f"\nDQN + UncertaintyVeto ranks {rank}/{total} in performance.")
        
        # Compare to simpler alternatives
        dqn_baseline = df[(df['agent'] == 'DQN') & (df['veto'] == 'NoVeto')]
        
        if not dqn_baseline.empty:
            main_reward = main_approach['avg_reward'].iloc[0]
            baseline_reward = dqn_baseline['avg_reward'].iloc[0]
            
            if main_reward > baseline_reward:
                improvement = ((main_reward - baseline_reward) / baseline_reward) * 100
                report.append(f"\nUncertaintyVeto improves DQN performance by {improvement:.1f}% compared to no veto.")
            else:
                difference = ((baseline_reward - main_reward) / baseline_reward) * 100
                report.append(f"\nUncertaintyVeto decreases DQN performance by {difference:.1f}% compared to no veto.")
                
        # Check if other veto mechanisms work better with DQN
        dqn_data = df[df['agent'] == 'DQN']
        best_veto_for_dqn = dqn_data.loc[dqn_data['avg_reward'].idxmax()]
        
        if best_veto_for_dqn['veto'] != 'UncertaintyVeto':
            report.append(f"\nFor DQN, {best_veto_for_dqn['veto']} performs better than UncertaintyVeto "
                        f"by {best_veto_for_dqn['avg_reward'] - main_approach['avg_reward'].iloc[0]:.2f} reward.")
            report.append("\nRecommendation: Consider using the simpler veto mechanism instead.")
        else:
            report.append("\nUncertaintyVeto is the best veto mechanism for DQN, confirming our approach.")
            
        # Check if the complexity is justified
        rule_based = df[(df['agent'] == 'RuleBased') & (df['veto'] == 'FixedRules')]
        
        if not rule_based.empty:
            simpler_reward = rule_based['avg_reward'].iloc[0]
            complex_reward = main_approach['avg_reward'].iloc[0]
            
            if complex_reward > simpler_reward:
                improvement = ((complex_reward - simpler_reward) / simpler_reward) * 100
                report.append(f"\nDQN + UncertaintyVeto improves performance by {improvement:.1f}% "
                           f"compared to the much simpler RuleBased + FixedRules approach.")
                
                if improvement < 10:
                    report.append("\nRecommendation: The small improvement may not justify the added complexity.")
                else:
                    report.append("\nRecommendation: The significant improvement justifies the added complexity.")
            else:
                report.append("\nThe simpler RuleBased + FixedRules approach outperforms DQN + UncertaintyVeto.")
                report.append("\nRecommendation: Use the simpler approach as it performs better with less complexity.")
    
    # Write report to file
    report_path = os.path.join(save_dir, 'benchmark_analysis_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--results', type=str, default='data/benchmark_results/benchmark_results.json', 
                      help='Path to benchmark results JSON file')
    parser.add_argument('--output', type=str, default='data/benchmark_results/analysis', 
                      help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not os.path.exists(args.results):
        print(f"Error: Results file {args.results} not found")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load benchmark results
    print(f"Loading benchmark results from {args.results}")
    df = load_benchmark_results(args.results)
    
    # Perform analysis
    print("Performing statistical analysis...")
    statistical_tests = perform_statistical_tests(df)
    
    print("Analyzing veto effectiveness...")
    veto_effectiveness = analyze_veto_effectiveness(df)
    
    analysis_results = {
        'statistical_tests': statistical_tests,
        'veto_effectiveness': veto_effectiveness
    }
    
    # Create visualizations
    print("Generating visualizations...")
    visualize_analysis(
        veto_effectiveness['df_with_efficiency'], 
        analysis_results, 
        args.output
    )
    
    # Generate report
    print("Generating analysis report...")
    report_path = generate_report(df, analysis_results, args.output)
    
    print(f"Analysis complete. Report saved to {report_path}")
    print(f"Visualizations saved to {args.output}")

if __name__ == "__main__":
    main()