"""
Analysis module for Veto Game experiments.
Provides tools for analyzing and visualizing experiment results.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class ExperimentAnalysis:
    """Class for analyzing experiment results"""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.sessions_df = None
        self.vetos_df = None
        self.actions_df = None
        
        # Load data
        self.load_data()
        
        # Create analysis directory
        self.analysis_dir = os.path.join(data_dir, "analysis")
        os.makedirs(self.analysis_dir, exist_ok=True)
        
    def load_data(self):
        """Load all session data into dataframes"""
        sessions = []
        vetos = []
        actions = []
        
        # Iterate through all files in the data directory
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json") and filename.startswith("p"):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path, 'r') as f:
                    try:
                        session_data = json.load(f)
                        
                        # Extract session ID from filename if not present
                        if "session_id" not in session_data:
                            # Use filename without extension as session ID
                            session_id = os.path.splitext(filename)[0]
                            session_data["session_id"] = session_id
                        
                        # Session data
                        if "participant_id" in session_data:
                            sessions.append({
                                "session_id": session_data["session_id"],
                                "participant_id": session_data["participant_id"],
                                "condition": session_data["condition"],
                                "start_time": session_data["start_time"],
                                "end_time": session_data["end_time"],
                                "duration": session_data["duration"] if "duration" in session_data else 
                                            (session_data["end_time"] - session_data["start_time"])
                            })
                        
                        # Veto data - check for both 'vetos' and 'veto_decisions'
                        veto_key = None
                        if "vetos" in session_data:
                            veto_key = "vetos"
                        elif "veto_decisions" in session_data:
                            veto_key = "veto_decisions"
                            
                        if veto_key and session_data[veto_key]:
                            for veto in session_data[veto_key]:
                                veto_entry = veto.copy()
                                veto_entry["session_id"] = session_data["session_id"]
                                veto_entry["participant_id"] = session_data["participant_id"]
                                veto_entry["condition"] = session_data["condition"]
                                vetos.append(veto_entry)
                        
                        # Action data
                        if "actions" in session_data and session_data["actions"]:
                            for action in session_data["actions"]:
                                # Only include actions with reward data
                                if "reward" in action:
                                    action_entry = action.copy()
                                    action_entry["session_id"] = session_data["session_id"]
                                    action_entry["participant_id"] = session_data["participant_id"]
                                    action_entry["condition"] = session_data["condition"]
                                    actions.append(action_entry)
                                
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse {filename}")
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
        
        # Create dataframes
        self.sessions_df = pd.DataFrame(sessions) if sessions else None
        self.vetos_df = pd.DataFrame(vetos) if vetos else None
        self.actions_df = pd.DataFrame(actions) if actions else None
        
        # Print some statistics for debugging
        print(f"Loaded {len(sessions)} sessions, {len(vetos)} vetos, {len(actions)} actions")
        
    def analyze_performance_by_condition(self):
        """Analyze performance metrics by condition"""
        if self.sessions_df is None:
            return {"error": "No session data available"}
            
        # Group by condition
        results = {}
        
        if self.actions_df is not None:
            # Calculate scores by condition
            condition_metrics = self.actions_df.groupby(['condition']).agg({
                'reward': ['mean', 'std', 'count']
            })
            
            results["condition_metrics"] = condition_metrics
            
            # Generate plot
            if len(condition_metrics) > 0:
                plt.figure(figsize=(10, 6))
                condition_metrics['reward']['mean'].plot(kind='bar', yerr=condition_metrics['reward']['std'], capsize=4)
                plt.title('Average Reward by Condition')
                plt.ylabel('Mean Reward')
                plt.tight_layout()
                plt.savefig(os.path.join(self.analysis_dir, 'reward_by_condition.png'))
                plt.close()
                
        return results
        
    def analyze_veto_outcomes(self):
        """Analyze outcomes of veto decisions"""
        if self.vetos_df is None:
            return {"error": "No veto data available"}
            
        results = {}
        
        # Check if we have reward data for vetos
        if 'reward' in self.vetos_df.columns:
            # Extract veto decisions where we have outcome data
            veto_outcomes = self.vetos_df.dropna(subset=['reward'])
            
            if len(veto_outcomes) > 0:
                # Group by vetoed (True/False)
                mean_rewards = veto_outcomes.groupby('vetoed')['reward'].agg(['mean', 'std', 'count'])
                results["mean_rewards"] = mean_rewards
                
                # Statistical test
                vetoed = veto_outcomes[veto_outcomes['vetoed'] == True]['reward']
                not_vetoed = veto_outcomes[veto_outcomes['vetoed'] == False]['reward']
                
                if len(vetoed) > 0 and len(not_vetoed) > 0:
                    t_stat, p_value = stats.ttest_ind(vetoed, not_vetoed, equal_var=False)
                    results["stats_result"] = {
                        "t_statistic": t_stat,
                        "p_value": p_value
                    }
                    
                # Generate plot
                plt.figure(figsize=(8, 6))
                mean_rewards['mean'].plot(kind='bar', yerr=mean_rewards['std'], capsize=4)
                plt.title('Reward Outcomes Based on Veto Decisions')
                plt.ylabel('Mean Reward')
                plt.xlabel('Vetoed')
                plt.xticks([0, 1], ['No', 'Yes'])
                plt.tight_layout()
                plt.savefig(os.path.join(self.analysis_dir, 'veto_outcomes.png'))
                plt.close()
        else:
            # Just provide veto decision counts
            if 'vetoed' in self.vetos_df.columns:
                veto_counts = self.vetos_df.groupby('vetoed').size()
                results["veto_counts"] = veto_counts
                
                # Generate pie chart
                plt.figure(figsize=(8, 6))
                labels = ['Accepted', 'Vetoed']
                sizes = [
                    veto_counts.get(False, 0), 
                    veto_counts.get(True, 0)
                ]
                plt.pie(sizes, labels=labels, autopct='%1.1f%%')
                plt.title('Veto Decision Distribution')
                plt.savefig(os.path.join(self.analysis_dir, 'veto_distribution.png'))
                plt.close()
                
            results["note"] = "No reward data available for veto outcomes analysis"
            
        return results
        
    def analyze_decision_times(self):
        """Analyze decision times for different conditions"""
        if self.vetos_df is None:
            return {"error": "No veto data available"}
            
        results = {}
        
        # Check if we have response time data
        response_time_field = None
        if 'response_time' in self.vetos_df.columns:
            response_time_field = 'response_time'
        elif 'decision_time' in self.vetos_df.columns:
            response_time_field = 'decision_time'
            
        if response_time_field:
            # Filter only veto requests with response time data
            decision_times = self.vetos_df.dropna(subset=[response_time_field])
            
            if len(decision_times) > 0:
                # Group by condition
                condition_times = decision_times.groupby('condition')[response_time_field].agg(['mean', 'std', 'count'])
                results["condition_times"] = condition_times
                
                # Generate plot
                plt.figure(figsize=(10, 6))
                condition_times['mean'].plot(kind='bar', yerr=condition_times['std'], capsize=4)
                plt.title('Decision Time by Condition')
                plt.ylabel('Mean Decision Time (seconds)')
                plt.tight_layout()
                plt.savefig(os.path.join(self.analysis_dir, 'decision_times.png'))
                plt.close()
        else:
            results["note"] = "No response time data available for decision time analysis"
            
        return results
        
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        report = {
            "experiment_summary": {
                "sessions": len(self.sessions_df) if self.sessions_df is not None else 0,
                "participants": len(self.sessions_df['participant_id'].unique()) if self.sessions_df is not None else 0,
                "conditions": list(self.sessions_df['condition'].unique()) if self.sessions_df is not None else [],
                "veto_decisions": len(self.vetos_df) if self.vetos_df is not None else 0,
                "actions": len(self.actions_df) if self.actions_df is not None else 0
            },
            "performance_by_condition": self.analyze_performance_by_condition(),
            "veto_outcomes": self.analyze_veto_outcomes(),
            "decision_times": self.analyze_decision_times()
        }
        
        # Save report as JSON
        report_path = os.path.join(self.analysis_dir, "analysis_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        return report