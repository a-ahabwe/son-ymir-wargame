import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class VetoMetrics:
    """Analyzes veto mechanism performance"""
    def __init__(self, save_dir='data/experiment_results'):
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Metrics
        self.veto_counts = {
            'total_actions': 0,
            'veto_requests': 0,
            'veto_approvals': 0,
            'veto_rejections': 0,
            'veto_timeouts': 0
        }
        
        # Timing metrics
        self.veto_response_times = []
        
        # Outcome metrics
        self.outcomes = {
            'approved_outcome': [],  # Reward outcomes of approved actions
            'rejected_outcome': []   # Reward outcomes of rejected actions
        }
        
    def record_action(self, veto_requested=False, veto_result=None, response_time=None, outcome=None):
        """Record an action for metrics"""
        self.veto_counts['total_actions'] += 1
        
        if veto_requested:
            self.veto_counts['veto_requests'] += 1
            
            if veto_result is True:  # Approved
                self.veto_counts['veto_approvals'] += 1
                if outcome is not None:
                    self.outcomes['approved_outcome'].append(outcome)
            elif veto_result is False:  # Rejected
                self.veto_counts['veto_rejections'] += 1
                if outcome is not None:
                    self.outcomes['rejected_outcome'].append(outcome)
            else:  # Timeout (None)
                self.veto_counts['veto_timeouts'] += 1
                
            if response_time is not None:
                self.veto_response_times.append(response_time)
                
    def calculate_metrics(self):
        """Calculate summary metrics"""
        metrics = {
            'counts': self.veto_counts.copy(),
            'veto_rate': self.veto_counts['veto_requests'] / max(1, self.veto_counts['total_actions']),
            'rejection_rate': self.veto_counts['veto_rejections'] / max(1, self.veto_counts['veto_requests']),
            'timeout_rate': self.veto_counts['veto_timeouts'] / max(1, self.veto_counts['veto_requests'])
        }
        
        # Response time metrics
        if self.veto_response_times:
            metrics['response_time'] = {
                'mean': np.mean(self.veto_response_times),
                'median': np.median(self.veto_response_times),
                'min': np.min(self.veto_response_times),
                'max': np.max(self.veto_response_times)
            }
            
        # Outcome metrics
        if self.outcomes['approved_outcome'] and self.outcomes['rejected_outcome']:
            metrics['outcome'] = {
                'mean_approved_outcome': np.mean(self.outcomes['approved_outcome']),
                'mean_rejected_outcome': np.mean(self.outcomes['rejected_outcome']),
                'outcome_difference': np.mean(self.outcomes['rejected_outcome']) - np.mean(self.outcomes['approved_outcome'])
            }
            
        return metrics
        
    def generate_plots(self, experiment_id=None):
        """Generate plots for visualization"""
        if experiment_id is None:
            experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Create experiment directory
        plot_dir = f"{self.save_dir}/{experiment_id}/plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
        # Plot 1: Veto counts
        plt.figure(figsize=(10, 6))
        labels = ['Total Actions', 'Veto Requests', 'Approvals', 'Rejections', 'Timeouts']
        values = [
            self.veto_counts['total_actions'],
            self.veto_counts['veto_requests'],
            self.veto_counts['veto_approvals'],
            self.veto_counts['veto_rejections'],
            self.veto_counts['veto_timeouts']
        ]
        plt.bar(labels, values)
        plt.title('Veto Counts')
        plt.ylabel('Count')
        plt.savefig(f"{plot_dir}/veto_counts.png")
        
        # Plot 2: Response time histogram
        if self.veto_response_times:
            plt.figure(figsize=(10, 6))
            plt.hist(self.veto_response_times, bins=20)
            plt.title('Veto Response Times')
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Frequency')
            plt.savefig(f"{plot_dir}/response_times.png")
            
        # Plot 3: Outcome comparison
        if self.outcomes['approved_outcome'] and self.outcomes['rejected_outcome']:
            plt.figure(figsize=(10, 6))
            plt.boxplot([self.outcomes['approved_outcome'], self.outcomes['rejected_outcome']])
            plt.xticks([1, 2], ['Approved', 'Rejected'])
            plt.title('Action Outcomes')
            plt.ylabel('Reward')
            plt.savefig(f"{plot_dir}/outcomes.png")
            
        return plot_dir
        
    def save_metrics(self, experiment_id=None):
        """Save metrics to file"""
        if experiment_id is None:
            experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Create experiment directory
        save_path = f"{self.save_dir}/{experiment_id}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Save to file
        import json
        with open(f"{save_path}/veto_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Generate plots
        plot_dir = self.generate_plots(experiment_id)
        
        return save_path