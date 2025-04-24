#!/usr/bin/env python3
"""
Improved experiment runner with better experimental design.
This version includes:
- Proper counterbalancing
- Standardized conditions
- A/B testing capabilities
- Better simulation of human veto behavior
- Statistical power analysis
"""

import argparse
import os
import time
import random
import sys
import json
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the src package is in the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.game.environment import GameEnvironment
from src.experiment.scenarios import ScenarioGenerator
from src.experiment.data_collection import ExperimentLogger
from src.ai.agent import RLAgent
from src.ai.uncertainty import UncertaintyEstimator
from src.veto.veto_mechanism import ThresholdVetoMechanism, UncertaintyVetoMechanism, VetoMechanism


class ImprovedExperiment:
    """
    Improved experiment class with better experimental design.
    """
    def __init__(self, experiment_id=None, output_dir="data/experiment_results", seed=None):
        """Initialize the experiment"""
        self.experiment_id = experiment_id or self._create_experiment_id()
        self.output_dir = os.path.join(output_dir, self.experiment_id)
        self.seed = seed
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = ExperimentLogger(self.output_dir)
        
        # Initialize components
        self.env = None
        self.agent = None
        self.uncertainty_estimator = None
        self.scenarios = []
        self.participants = []
        self.conditions = []
        
        # Experiment configuration
        self.config = {
            "experiment_id": self.experiment_id,
            "start_time": time.time(),
            "seed": seed,
            "hypotheses": [],
            "conditions": [],
            "participants": [],
            "scenarios": []
        }
        
        # Statistical power analysis settings
        self.power_analysis = {
            "target_power": 0.8,  # Standard target power (80%)
            "alpha": 0.05,        # Significance level
            "effect_size": 0.5,   # Medium effect size
            "estimated_variance": 1.0  # Initial estimate
        }
        
        # Veto behavior simulation parameters for virtual participants
        self.veto_behavior_models = {
            "conservative": {
                "base_veto_probability": 0.7,
                "risk_sensitivity": 0.8,
                "uncertainty_sensitivity": 0.6,
                "consistency": 0.8
            },
            "moderate": {
                "base_veto_probability": 0.5,
                "risk_sensitivity": 0.5,
                "uncertainty_sensitivity": 0.5,
                "consistency": 0.6
            },
            "permissive": {
                "base_veto_probability": 0.3,
                "risk_sensitivity": 0.3,
                "uncertainty_sensitivity": 0.4,
                "consistency": 0.4
            }
        }
        
    def _create_experiment_id(self):
        """Create a unique experiment ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"experiment_{timestamp}"
        
    def add_hypothesis(self, hypothesis, null_hypothesis=None, type="superiority", 
                      min_effect_size=0.3, alpha=0.05):
        """
        Add a formal hypothesis to test in the experiment
        
        Args:
            hypothesis (str): The alternative hypothesis statement
            null_hypothesis (str): The null hypothesis statement (default: inverse of hypothesis)
            type (str): Type of hypothesis test - 'superiority', 'non-inferiority', or 'equivalence'
            min_effect_size (float): Minimum effect size considered meaningful
            alpha (float): Significance level for hypothesis testing
        """
        hypothesis_id = len(self.config["hypotheses"]) + 1
        
        if null_hypothesis is None:
            null_hypothesis = f"No difference or the opposite of hypothesis {hypothesis_id}"
            
        self.config["hypotheses"].append({
            "id": hypothesis_id,
            "hypothesis": hypothesis,
            "null_hypothesis": null_hypothesis,
            "type": type,
            "min_effect_size": min_effect_size,
            "alpha": alpha,
            "result": None  # Will be filled after analysis
        })
        
        return hypothesis_id
        
    def setup_conditions(self, conditions, balanced=True, include_control=True):
        """
        Set up experimental conditions
        
        Args:
            conditions (list): List of condition names or dictionaries
            balanced (bool): Whether to ensure a balanced design
            include_control (bool): Whether to include a control condition
        """
        self.conditions = []
        
        # Process conditions
        for condition in conditions:
            if isinstance(condition, str):
                condition = {"name": condition}
                
            self.conditions.append(condition)
            
        # Add control condition if requested
        if include_control and not any(c.get("name") == "control" for c in self.conditions):
            self.conditions.append({"name": "control", "is_control": True})
            
        # Save to config
        self.config["conditions"] = self.conditions
        
        return self.conditions
        
    def setup_scenarios(self, num_scenarios=5, difficulty_range=(0.1, 0.9)):
        """Create standardized test scenarios with controlled difficulty levels"""
        # Generate fixed seed for reproducibility
        scenario_seed = self.seed if self.seed is not None else 42
        
        # Create scenario generator
        generator = ScenarioGenerator(seed=scenario_seed)
        
        # Generate scenarios with fixed difficulty levels
        min_diff, max_diff = difficulty_range
        difficulties = np.linspace(min_diff, max_diff, num_scenarios)
        
        # Generate scenarios with default names
        self.scenarios = generator.generate_scenarios(num_scenarios)
        
        # Save to config
        self.config["scenarios"] = [
            {"name": s.name, "difficulty": s.difficulty} for s in self.scenarios
        ]
        
        return self.scenarios
        
    def setup_participants(self, num_participants=30, virtual=True, balanced=True):
        """
        Set up participants with counterbalancing
        
        Args:
            num_participants (int): Number of participants
            virtual (bool): Whether to use simulated participants
            balanced (bool): Whether to ensure a balanced design
        """
        self.participants = []
        
        # Create participant profiles
        for i in range(1, num_participants + 1):
            participant = {
                "id": i,
                "virtual": virtual,
                # Assign one of three veto behavior models for variety
                "veto_behavior": random.choice(list(self.veto_behavior_models.keys()))
            }
            
            # Counterbalance conditions
            if balanced and self.conditions:
                # Create balanced blocks of condition orders
                num_conditions = len(self.conditions)
                all_orders = list(itertools.permutations(range(num_conditions)))
                
                # Select a balanced order for this participant
                order_idx = (i - 1) % len(all_orders)
                condition_order = all_orders[order_idx]
                
                # Assign ordered conditions
                participant["condition_order"] = [
                    self.conditions[idx]["name"] for idx in condition_order
                ]
            else:
                # Random order
                participant["condition_order"] = [c["name"] for c in self.conditions]
                random.shuffle(participant["condition_order"])
                
            self.participants.append(participant)
            
        # Save to config
        self.config["participants"] = self.participants
        
        # Run power analysis to check if we have enough participants
        self._run_power_analysis(num_participants)
        
        return self.participants
        
    def _run_power_analysis(self, num_participants):
        """Run a power analysis to determine if we have enough participants"""
        # Calculate estimated power for current sample size
        from statsmodels.stats.power import TTestIndPower
        
        # Initialize power analysis
        power_analysis = TTestIndPower()
        
        # Check if sample size is sufficient for power analysis (requires at least 1 per group)
        if num_participants < 2:
            print(f"Warning: Not enough participants ({num_participants}) for power analysis. Skipping.")
            power = np.nan
            sample_size = np.nan
        else:
            nobs1 = num_participants // 2 
            # Ensure nobs1 is at least 1, although num_participants < 2 check should cover this
            if nobs1 == 0: 
                 nobs1 = 1 # Should not happen if num_participants >= 2
            
            # Calculate power
            power = power_analysis.solve_power(
                effect_size=self.power_analysis["effect_size"],
                nobs1=nobs1,  # Use calculated nobs1
                alpha=self.power_analysis["alpha"]
            )
            
            # Calculate required sample size for target power
            sample_size = power_analysis.solve_power(
                effect_size=self.power_analysis["effect_size"],
                power=self.power_analysis["target_power"],
                alpha=self.power_analysis["alpha"],
                ratio=1.0  # Equal group sizes
            )
        
        # Update power analysis results
        self.power_analysis["estimated_power"] = power
        self.power_analysis["recommended_sample_size"] = int(np.ceil(sample_size * 2)) if not np.isnan(sample_size) else np.nan
        self.power_analysis["current_sample_size"] = num_participants
        
        # Save to config
        self.config["power_analysis"] = self.power_analysis
        
        return self.power_analysis
        
    def simulate_veto_decision(self, participant, state, action, q_values, veto_info):
        """
        Simulate human veto decision based on participant profile
        
        This provides a more realistic model of human behavior than simple random vetoes
        """
        # Get participant's veto behavior profile
        behavior = self.veto_behavior_models[participant["veto_behavior"]]
        
        # Extract relevant information
        base_prob = behavior["base_veto_probability"]
        risk_sensitivity = behavior["risk_sensitivity"]
        uncertainty_sensitivity = behavior["uncertainty_sensitivity"]
        consistency = behavior["consistency"]
        
        # Extract veto information
        is_risky = veto_info.get("is_risky", False)
        risk_level = veto_info.get("risk_level", 0.0)
        uncertainty = veto_info.get("uncertainty", 0.0)
        
        # Base probability of vetoing
        veto_probability = base_prob
        
        # Adjust based on risk assessment
        if is_risky:
            veto_probability += risk_level * risk_sensitivity
            
        # Adjust based on uncertainty
        if uncertainty is not None and uncertainty > 0.5:
            veto_probability += (uncertainty - 0.5) * 2 * uncertainty_sensitivity
            
        # Adjust for Q-value difference if available
        if q_values is not None:
            best_action = np.argmax(q_values)
            best_q = q_values[best_action]
            action_q = q_values[action]
            
            if best_q > action_q and best_action != action:
                # The bigger the difference, the more likely to veto
                q_diff = (best_q - action_q) / max(1.0, best_q)
                veto_probability += q_diff * 0.3
                
        # Add some randomness (less randomness = more consistent behavior)
        randomness = 1.0 - consistency
        veto_probability = max(0.0, min(1.0, 
                                      veto_probability + random.uniform(-randomness, randomness)))
        
        # Make decision
        return random.random() < veto_probability
        
    def _setup_components(self):
        """Set up environment, agent, and other components"""
        # Create environment
        self.env = GameEnvironment(grid_size=50, seed=self.seed)
        
        # Create agent
        self.agent = RLAgent(self.env.state_size, self.env.action_space_n)
        
        # Create uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(self.agent.model)
        
    def run_experiment(self):
        """Run the full experiment"""
        start_time = time.time()
        print(f"Starting experiment {self.experiment_id}")
        
        # Set up components if not already done
        if self.env is None:
            self._setup_components()
            
        # Ensure we have conditions and scenarios
        if not self.conditions:
            self.setup_conditions(["threshold", "uncertainty"])
            
        if not self.scenarios:
            self.setup_scenarios()
            
        if not self.participants:
            self.setup_participants()
            
        # Save initial configuration
        self._save_config()
        
        # Run for each participant
        for participant in self.participants:
            print(f"Running experiment for participant {participant['id']}")
            
            # Run participant through all conditions in their assigned order
            for condition_name in participant["condition_order"]:
                # Create appropriate veto mechanism
                veto_mechanism = self._create_veto_mechanism(condition_name)
                
                # Start session
                self.logger.start_session(participant["id"], condition_name)
                print(f"  Condition: {condition_name}")
                
                # Run each scenario
                for scenario in self.scenarios:
                    print(f"    Scenario: {scenario.name}")
                    
                    # Apply scenario to environment
                    env = scenario.apply_to_environment(self.env)
                    
                    # Run scenario
                    self._run_scenario(participant, env, veto_mechanism, scenario)
                    
                # End session
                self.logger.end_session()
                
        # Record experiment end time
        end_time = time.time()
        self.config["end_time"] = end_time
        self.config["duration"] = end_time - start_time
        
        # Save final configuration
        self._save_config()
        
        print(f"Experiment {self.experiment_id} completed in {end_time - start_time:.1f} seconds")
        
        return self.output_dir
        
    def _create_veto_mechanism(self, condition_name):
        """Create appropriate veto mechanism for the condition"""
        if condition_name == "threshold":
            return ThresholdVetoMechanism(threshold=0.7, timeout=10)
        elif condition_name == "uncertainty":
            return UncertaintyVetoMechanism(
                self.uncertainty_estimator, 
                uncertainty_threshold=0.5,
                timeout=10
            )
        elif condition_name == "control":
            # Control condition uses base veto mechanism that never vetoes
            return VetoMechanism(timeout=10)
        else:
            # Default to threshold mechanism
            return ThresholdVetoMechanism(threshold=0.7, timeout=10)
            
    def _run_scenario(self, participant, env, veto_mechanism, scenario, max_steps=200):
        """Run a single scenario for a participant"""
        # Reset environment
        state = env.reset()
        
        # Track metrics
        total_reward = 0
        total_steps = 0
        veto_count = 0
        successful_vetos = 0
        
        # Run for maximum number of steps
        for step in range(max_steps):
            # Select action using AI
            action, q_values = self.agent.select_action(state)
            
            # Check for veto
            veto_decision = veto_mechanism.assess_action(state, action, q_values)
            veto_info = {
                "is_risky": veto_decision.vetoed,
                "risk_reason": veto_decision.reason,
                "uncertainty": getattr(veto_decision, "uncertainty", 0.0),
                "risk_level": 0.5  # Default risk level
            }
            
            # For virtual participants, simulate veto decision
            if participant["virtual"]:
                # Simulate human decision
                human_would_veto = self.simulate_veto_decision(
                    participant, state, action, q_values, veto_info
                )
                
                if veto_decision.vetoed:
                    # Log veto request
                    request_id = self.logger.log_veto_request({
                        'action': action,
                        'action_desc': self._get_action_description(action),
                        'risk_reason': veto_decision.reason,
                        'q_values': q_values.tolist() if hasattr(q_values, 'tolist') else q_values
                    })
                    
                    if human_would_veto:
                        veto_count += 1
                        
                        # Choose alternative action
                        alternative_action = self.agent.select_safe_action(state, action)
                        
                        # Execute alternative action
                        next_state, reward, done, _ = env.step(alternative_action)
                        
                        # Log veto decision
                        self.logger.log_veto({
                            'request_id': request_id,
                            'original_action': action,
                            'vetoed': True,
                            'alternative': alternative_action,
                            'reward': reward
                        })
                        
                        # Track successful vetos
                        if reward > 0:
                            successful_vetos += 1
                    else:
                        # Execute original action
                        next_state, reward, done, _ = env.step(action)
                        
                        # Log veto decision
                        self.logger.log_veto({
                            'request_id': request_id,
                            'original_action': action,
                            'vetoed': False,
                            'alternative': None,
                            'reward': reward
                        })
                else:
                    # Execute original action
                    next_state, reward, done, _ = env.step(action)
                    
                    # Log action
                    self.logger.log_action({
                        'action': action,
                        'reward': reward,
                        'q_value': q_values[action] if q_values is not None else None
                    })
            else:
                # In a real experiment with actual humans, this would wait for UI input
                # For now, just execute the action without veto
                next_state, reward, done, _ = env.step(action)
                
                # Log action
                self.logger.log_action({
                    'action': action,
                    'reward': reward,
                    'q_value': q_values[action] if q_values is not None else None
                })
            
            # Update state and metrics
            state = next_state
            total_reward += reward
            total_steps += 1
            
            if done:
                break
                
        # Log scenario results
        scenario_results = {
            'participant_id': participant["id"],
            'scenario': scenario.name,
            'total_reward': total_reward,
            'total_steps': total_steps,
            'veto_count': veto_count,
            'successful_vetos': successful_vetos
        }
        
        # In a real implementation, we'd log this to the experiment logger
        print(f"      Reward: {total_reward}, Steps: {total_steps}, Vetos: {veto_count}")
        
        return scenario_results
        
    def _get_action_description(self, action):
        """Get human-readable description of an action"""
        action_map = {
            0: "Move Up",
            1: "Move Down",
            2: "Move Left",
            3: "Move Right",
            4: "Shoot Up",
            5: "Shoot Down",
            6: "Shoot Left",
            7: "Shoot Right",
            8: "Place Trap",
            9: "Use Cover",
            10: "Call Support"
        }
        return action_map.get(action, f"Unknown Action {action}")
        
    def _save_config(self):
        """Save experiment configuration to file"""
        config_path = os.path.join(self.output_dir, "experiment_config.json")
        
        # Convert any non-serializable objects
        config_copy = self._make_json_serializable(self.config)
        
        with open(config_path, 'w') as f:
            json.dump(config_copy, f, indent=2)
            
    def _make_json_serializable(self, obj):
        """Convert any non-serializable objects to serializable ones"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        else:
            return obj
            
    def analyze_results(self):
        """Analyze experiment results and test hypotheses"""
        # Load and organize data
        print(f"Analyzing results for experiment {self.experiment_id}")
        
        # Load session data
        sessions = []
        vetos = []
        actions = []
        
        # Iterate through JSON files in output directory
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.json') and filename.startswith('p'):
                file_path = os.path.join(self.output_dir, filename)
                with open(file_path, 'r') as f:
                    try:
                        session_data = json.load(f)
                        
                        # Session info
                        session_info = {
                            "participant_id": session_data["participant_id"],
                            "condition": session_data["condition"],
                            "start_time": session_data["start_time"],
                            "end_time": session_data["end_time"],
                            "duration": session_data.get("duration", 
                                                      session_data["end_time"] - session_data["start_time"])
                        }
                        sessions.append(session_info)
                        
                        # Veto data
                        for veto in session_data.get("veto_decisions", []):
                            veto_copy = veto.copy()
                            veto_copy["participant_id"] = session_data["participant_id"]
                            veto_copy["condition"] = session_data["condition"]
                            vetos.append(veto_copy)
                            
                        # Action data
                        for action in session_data.get("actions", []):
                            action_copy = action.copy()
                            action_copy["participant_id"] = session_data["participant_id"]
                            action_copy["condition"] = session_data["condition"]
                            actions.append(action_copy)
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
                        
        # Convert to DataFrames
        sessions_df = pd.DataFrame(sessions)
        vetos_df = pd.DataFrame(vetos)
        actions_df = pd.DataFrame(actions)
        
        # Create analysis directory
        analysis_dir = os.path.join(self.output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Compare conditions
        analysis_results = {}
        
        if not sessions_df.empty:
            # 1. Overall metrics by condition
            condition_metrics = self._analyze_condition_metrics(actions_df, vetos_df)
            analysis_results["condition_metrics"] = condition_metrics
            
            # 2. Veto effectiveness
            veto_effectiveness = self._analyze_veto_effectiveness(vetos_df)
            analysis_results["veto_effectiveness"] = veto_effectiveness
            
            # 3. Test hypotheses
            hypothesis_results = self._test_hypotheses(actions_df, vetos_df)
            analysis_results["hypothesis_results"] = hypothesis_results
            
            # Update config with hypothesis results
            for i, hypothesis in enumerate(self.config["hypotheses"]):
                if i < len(hypothesis_results):
                    hypothesis["result"] = hypothesis_results[i]
                    
            # Save updated config
            self._save_config()
            
            # 4. Generate visualizations
            self._generate_analysis_visualizations(
                actions_df, vetos_df, condition_metrics, analysis_dir
            )
            
            # 5. Save analysis results
            results_path = os.path.join(analysis_dir, "analysis_results.json")
            with open(results_path, 'w') as f:
                json.dump(self._make_json_serializable(analysis_results), f, indent=2)
                
            print(f"Analysis results saved to {results_path}")
            
        return analysis_results
        
    def _analyze_condition_metrics(self, actions_df, vetos_df):
        """Calculate metrics by condition"""
        metrics = {}
        
        if not actions_df.empty:
            # Calculate average reward by condition
            reward_by_condition = actions_df.groupby('condition')['reward'].agg(['mean', 'std', 'count'])
            metrics["reward_by_condition"] = reward_by_condition.to_dict()
            
        if not vetos_df.empty:
            # Calculate veto statistics by condition
            veto_counts = vetos_df.groupby(['condition', 'vetoed']).size().unstack(fill_value=0)
            
            # Add derived metrics
            if 'True' in veto_counts.columns and 'False' in veto_counts.columns:
                veto_counts['total_requests'] = veto_counts['True'] + veto_counts['False']
                veto_counts['veto_rate'] = veto_counts['True'] / veto_counts['total_requests']
                
            metrics["veto_counts"] = veto_counts.to_dict()
            
            # Calculate effect of veto on reward
            if 'reward' in vetos_df.columns:
                veto_rewards = vetos_df.groupby('vetoed')['reward'].agg(['mean', 'std', 'count'])
                metrics["veto_rewards"] = veto_rewards.to_dict()
                
        return metrics
        
    def _analyze_veto_effectiveness(self, vetos_df):
        """Analyze the effectiveness of veto decisions"""
        results = {}
        
        if not vetos_df.empty and 'reward' in vetos_df.columns:
            # Calculate average reward for vetoed vs. non-vetoed actions
            veto_rewards = vetos_df.groupby('vetoed')['reward'].agg(['mean', 'std', 'count']).to_dict()
            results["reward_by_veto_decision"] = veto_rewards
            
            # Statistical test
            try:
                vetoed = vetos_df[vetos_df['vetoed'] == True]['reward']
                not_vetoed = vetos_df[vetos_df['vetoed'] == False]['reward']
                
                if len(vetoed) > 0 and len(not_vetoed) > 0:
                    t_stat, p_value = stats.ttest_ind(vetoed, not_vetoed, equal_var=False)
                    
                    results["statistical_test"] = {
                        "test": "t-test",
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05
                    }
            except Exception as e:
                print(f"Error in veto effectiveness analysis: {str(e)}")
                
        return results
        
    def _test_hypotheses(self, actions_df, vetos_df):
        """Test the formal hypotheses defined for the experiment"""
        results = []
        
        # For each hypothesis in config
        for hypothesis in self.config["hypotheses"]:
            result = {
                "hypothesis_id": hypothesis["id"],
                "result": "not_tested",
                "p_value": None,
                "effect_size": None,
                "significant": False
            }
            
            # Example: If hypothesis is about uncertainty veto being better than threshold
            if "uncertainty" in hypothesis["hypothesis"].lower() and "threshold" in hypothesis["hypothesis"].lower():
                if not actions_df.empty:
                    try:
                        # Extract reward data by condition
                        uncertainty_rewards = actions_df[actions_df['condition'] == 'uncertainty']['reward']
                        threshold_rewards = actions_df[actions_df['condition'] == 'threshold']['reward']
                        
                        if len(uncertainty_rewards) > 0 and len(threshold_rewards) > 0:
                            # Calculate effect size (Cohen's d)
                            effect_size = (uncertainty_rewards.mean() - threshold_rewards.mean()) / np.sqrt(
                                (uncertainty_rewards.var() + threshold_rewards.var()) / 2
                            )
                            
                            # Perform t-test
                            t_stat, p_value = stats.ttest_ind(
                                uncertainty_rewards, threshold_rewards, equal_var=False
                            )
                            
                            # Determine result
                            if p_value < hypothesis["alpha"]:
                                if effect_size > 0:
                                    test_result = "supported"
                                else:
                                    test_result = "contradicted"
                            else:
                                test_result = "not_significant"
                                
                            result.update({
                                "result": test_result,
                                "p_value": float(p_value),
                                "effect_size": float(effect_size),
                                "significant": p_value < hypothesis["alpha"],
                                "test_statistic": float(t_stat)
                            })
                    except Exception as e:
                        print(f"Error testing hypothesis {hypothesis['id']}: {str(e)}")
            
            results.append(result)
            
        return results
        
    def _generate_analysis_visualizations(self, actions_df, vetos_df, condition_metrics, output_dir):
        """Generate analysis visualizations"""
        # Calculate veto rate by condition first to avoid reference error
        if not vetos_df.empty:
            veto_rate_by_condition = vetos_df.groupby('condition')['vetoed'].mean()
        else:
            veto_rate_by_condition = pd.Series(dtype=float)

        # 1. Veto rate by condition
        if not vetos_df.empty and not veto_rate_by_condition.empty:
            plt.figure(figsize=(10, 6))
            veto_rate_by_condition.plot(kind='bar')
            plt.title("Veto Rate by Condition")
            plt.ylabel("Veto Rate")
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "veto_rate_by_condition.png"))
            plt.close()
            
        # 2. Reward by condition
        if "reward_by_condition" in condition_metrics and not actions_df.empty:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=actions_df, x="condition", y="reward", ci=68)
            plt.title("Average Reward by Condition")
            plt.ylabel("Reward")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "reward_by_condition.png"))
            plt.close()
            
        # 3. Veto effectiveness
        if not vetos_df.empty and 'reward' in vetos_df.columns:
            plt.figure(figsize=(8, 6))
            sns.barplot(data=vetos_df, x="vetoed", y="reward", ci=68)
            plt.title("Reward by Veto Decision")
            plt.xlabel("Vetoed")
            plt.ylabel("Reward")
            plt.xticks([0, 1], ["No", "Yes"])
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "veto_effectiveness.png"))
            plt.close()
            
        # 4. Distribution of rewards by condition
        if not actions_df.empty:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=actions_df, x="condition", y="reward")
            plt.title("Distribution of Rewards by Condition")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "reward_distribution.png"))
            plt.close()
            
        # 5. Successful veto ratio by condition
        if not vetos_df.empty and 'reward' in vetos_df.columns:
            # Calculate successful veto ratio (vetoes that led to positive rewards)
            vetos_df['successful'] = (vetos_df['vetoed'] == True) & (vetos_df['reward'] > 0)
            
            # Create summary
            successful_vetos = vetos_df[vetos_df['vetoed'] == True].groupby('condition')['successful'].mean()
            
            plt.figure(figsize=(10, 6))
            successful_vetos.plot(kind='bar')
            plt.title("Successful Veto Ratio by Condition")
            plt.ylabel("Success Rate")
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "successful_veto_ratio.png"))
            plt.close()