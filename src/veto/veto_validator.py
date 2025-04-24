"""
Validation module for veto decisions.
Compares veto mechanism decisions against ground truth.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from src.game.state import GameState
from src.veto.ground_truth import GroundTruthOracle

class VetoValidator:
    """
    Validator for veto mechanism decisions.
    Compares mechanism decisions with ground truth to evaluate performance.
    """
    def __init__(self, environment_class, oracle=None):
        """
        Initialize veto validator
        
        Args:
            environment_class: Class of the environment (not instance)
            oracle: Optional pre-configured GroundTruthOracle
        """
        self.environment_class = environment_class
        self.oracle = oracle or GroundTruthOracle(environment_class)
        
        # Tracking data for validation
        self.validation_data = []
        
        # Performance metrics
        self.performance = defaultdict(lambda: {
            'true_positives': 0,  # Correct vetoes
            'false_positives': 0, # Unnecessary vetoes
            'true_negatives': 0,  # Correct non-vetoes
            'false_negatives': 0, # Missed vetoes
            'total': 0
        })
    
    def validate_veto_decision(self, mechanism, state, action, q_values=None, record=True):
        """
        Validate a veto decision against ground truth
        
        Args:
            mechanism: Veto mechanism to validate
            state: GameState object or raw state array
            action: Action to evaluate
            q_values: Optional Q-values
            record: Whether to record this validation
            
        Returns:
            (correct, metrics, explanation): Validation results
        """
        # Convert to GameState if needed
        if not isinstance(state, GameState):
            state = GameState(raw_state=state)
            
        # Get mechanism's decision
        veto_decision = mechanism.assess_action(state, action, q_values)
        mechanism_vetoed = veto_decision.vetoed
        mechanism_reason = veto_decision.reason
        
        # Get ground truth decision
        gt_vetoed, gt_confidence, gt_explanation = self.oracle.should_veto(state, action, q_values)
        
        # Determine correctness
        correct = mechanism_vetoed == gt_vetoed
        
        # Calculate confidence and metrics
        if mechanism_vetoed and gt_vetoed:
            decision_type = 'true_positive'
        elif mechanism_vetoed and not gt_vetoed:
            decision_type = 'false_positive'
        elif not mechanism_vetoed and not gt_vetoed:
            decision_type = 'true_negative'
        else:  # not mechanism_vetoed and gt_vetoed
            decision_type = 'false_negative'
            
        # Update metrics
        mechanism_name = mechanism.__class__.__name__
        self.performance[mechanism_name][decision_type] += 1
        self.performance[mechanism_name]['total'] += 1
        
        # Record validation if requested
        if record:
            self.validation_data.append({
                'mechanism': mechanism_name,
                'action': action,
                'mechanism_vetoed': mechanism_vetoed,
                'ground_truth_vetoed': gt_vetoed,
                'correct': correct,
                'mechanism_reason': mechanism_reason,
                'ground_truth_explanation': gt_explanation,
                'decision_type': decision_type,
                'ground_truth_confidence': gt_confidence
            })
            
        # Create explanation
        if correct:
            explanation = f"Correct {decision_type} decision"
        else:
            explanation = f"Incorrect {decision_type} decision"
            
        # Create metrics
        metrics = {
            'correct': correct,
            'decision_type': decision_type,
            'ground_truth_confidence': gt_confidence
        }
        
        return correct, metrics, explanation
    
    def batch_validate(self, mechanism, states, actions, q_values_list=None):
        """
        Validate multiple decisions at once
        
        Args:
            mechanism: Veto mechanism to validate
            states: List of states
            actions: List of actions
            q_values_list: Optional list of Q-values
            
        Returns:
            Summary metrics
        """
        correct_count = 0
        
        # Process each decision
        for i, (state, action) in enumerate(zip(states, actions)):
            q_values = q_values_list[i] if q_values_list else None
            
            # Validate decision
            correct, _, _ = self.validate_veto_decision(mechanism, state, action, q_values)
            
            if correct:
                correct_count += 1
                
        # Calculate accuracy
        accuracy = correct_count / len(states) if states else 0
        
        return {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total': len(states)
        }
    
    def calculate_performance_metrics(self, mechanism_name=None):
        """
        Calculate detailed performance metrics for one or all mechanisms
        
        Args:
            mechanism_name: Optional mechanism name to filter for
            
        Returns:
            Dictionary of performance metrics
        """
        result = {}
        
        # Process data for selected mechanism(s)
        mechanisms = [mechanism_name] if mechanism_name else self.performance.keys()
        
        for name in mechanisms:
            if name not in self.performance:
                continue
                
            metrics = self.performance[name]
            
            # Calculate key metrics
            tp = metrics['true_positives']
            fp = metrics['false_positives']
            tn = metrics['true_negatives']
            fn = metrics['false_negatives']
            total = metrics['total']
            
            # Basic metrics
            accuracy = (tp + tn) / total if total > 0 else 0
            
            # Precision: TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Recall: TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # F1 score: 2 * (precision * recall) / (precision + recall)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Specificity: TN / (TN + FP)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Store results
            result[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn,
                'total': total
            }
            
        return result
    
    def generate_validation_report(self, output_path=None):
        """
        Generate a comprehensive validation report
        
        Args:
            output_path: Optional path to save report and visualizations
            
        Returns:
            Dictionary with report data
        """
        # Convert validation data to DataFrame for analysis
        df = pd.DataFrame(self.validation_data)
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics()
        
        # Create report data
        report = {
            'performance_metrics': performance,
            'oracle_metrics': self.oracle.get_metrics(),
            'action_statistics': self.oracle.get_action_statistics(),
            'validation_count': len(self.validation_data)
        }
        
        # Create visualizations if output path provided
        if output_path and not df.empty:
            self._create_performance_visualizations(df, performance, output_path)
            
        return report
    
    def _create_performance_visualizations(self, df, performance, output_path):
        """Create visualizations for validation report"""
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # 1. Confusion matrix for each mechanism
        for mechanism_name in df['mechanism'].unique():
            mechanism_df = df[df['mechanism'] == mechanism_name]
            
            # Create confusion matrix
            labels = ['Not Vetoed', 'Vetoed']
            cm = np.zeros((2, 2))
            
            # Fill confusion matrix: rows = mechanism decision, cols = ground truth
            cm[0, 0] = len(mechanism_df[(mechanism_df['mechanism_vetoed'] == False) & 
                                      (mechanism_df['ground_truth_vetoed'] == False)])
            cm[0, 1] = len(mechanism_df[(mechanism_df['mechanism_vetoed'] == False) & 
                                      (mechanism_df['ground_truth_vetoed'] == True)])
            cm[1, 0] = len(mechanism_df[(mechanism_df['mechanism_vetoed'] == True) & 
                                      (mechanism_df['ground_truth_vetoed'] == False)])
            cm[1, 1] = len(mechanism_df[(mechanism_df['mechanism_vetoed'] == True) & 
                                      (mechanism_df['ground_truth_vetoed'] == True)])
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix: {mechanism_name}')
            plt.colorbar()
            tick_marks = [0, 1]
            plt.xticks(tick_marks, labels, rotation=45)
            plt.yticks(tick_marks, labels)
            plt.xlabel('Ground Truth')
            plt.ylabel('Mechanism Decision')
            
            # Add text annotations
            thresh = cm.max() / 2
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, format(cm[i, j], 'd'),
                           horizontalalignment="center",
                           color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f'confusion_matrix_{mechanism_name}.png'))
            plt.close()
            
        # 2. Performance comparison
        if len(performance) > 1:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
            
            # Create comparison bar chart for each metric
            for metric in metrics:
                plt.figure(figsize=(10, 6))
                
                # Extract data
                names = list(performance.keys())
                values = [performance[name][metric] for name in names]
                
                # Create bar chart
                plt.bar(names, values)
                plt.title(f'{metric.replace("_", " ").title()} Comparison')
                plt.ylabel(metric.replace("_", " ").title())
                plt.ylim(0, 1)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, f'comparison_{metric}.png'))
                plt.close()