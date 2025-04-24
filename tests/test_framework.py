"""
Standardized testing framework for veto mechanisms.
Provides tools to validate and test veto components consistently.
"""

import unittest
import numpy as np
import torch
import time
import os
import json
from collections import defaultdict
from src.game.state import GameState
from src.veto.ground_truth import GroundTruthOracle
from src.veto.veto_validator import VetoValidator
from src.ai.improved_uncertainty import ImprovedUncertaintyEstimator

class VetoTestSuite:
    """
    Comprehensive test suite for veto mechanisms.
    Runs a standard set of tests to validate core functionality.
    """
    def __init__(self, environment_class, save_dir=None):
        """
        Initialize test suite
        
        Args:
            environment_class: Class of environment to use for testing
            save_dir: Directory to save test results
        """
        self.environment_class = environment_class
        self.save_dir = save_dir
        
        # Create directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # Create ground truth oracle for validation
        self.oracle = GroundTruthOracle(environment_class)
        
        # Create validator
        self.validator = VetoValidator(environment_class, self.oracle)
        
        # Test results
        self.results = {}
        
    def run_all_tests(self, mechanism, agent=None):
        """
        Run all tests for a veto mechanism
        
        Args:
            mechanism: Veto mechanism to test
            agent: Optional agent to use for action selection
            
        Returns:
            Dictionary with test results
        """
        mechanism_name = mechanism.__class__.__name__
        print(f"Running test suite for {mechanism_name}...")
        
        test_results = {}
        
        # Basic functionality tests
        print("  Testing basic functionality...")
        test_results['basic'] = self.test_basic_functionality(mechanism)
        
        # Consistency tests
        print("  Testing consistency...")
        test_results['consistency'] = self.test_consistency(mechanism)
        
        # Edge case tests
        print("  Testing edge cases...")
        test_results['edge_cases'] = self.test_edge_cases(mechanism)
        
        # Performance tests
        print("  Testing performance...")
        test_results['performance'] = self.test_performance(mechanism)
        
        # Integration tests with agent
        if agent:
            print("  Testing agent integration...")
            test_results['agent_integration'] = self.test_agent_integration(mechanism, agent)
            
        # Ground truth validation
        print("  Validating against ground truth...")
        test_results['ground_truth'] = self.validate_against_ground_truth(mechanism)
        
        # Store results
        self.results[mechanism_name] = test_results
        
        # Calculate overall score
        score = self._calculate_overall_score(test_results)
        test_results['overall_score'] = score
        
        print(f"  Overall score: {score:.2f} / 10.0")
        
        # Save results if directory specified
        if self.save_dir:
            self._save_results(mechanism_name, test_results)
            
        return test_results
    
    def test_basic_functionality(self, mechanism):
        """
        Test basic functionality of the veto mechanism
        
        Args:
            mechanism: Veto mechanism to test
            
        Returns:
            Dictionary with test results
        """
        results = {
            'passed': True,
            'failures': [],
            'details': {}
        }
        
        # Create test state with known properties
        state = self._create_test_state(health=0.5, ammo=0.5, shields=0.5)
        
        # Check that mechanism can process actions without error
        try:
            # Try all possible actions
            for action in range(11):  # Assuming 11 possible actions
                # Create some Q-values
                q_values = np.zeros(11)
                q_values[action] = 0.5  # Give this action medium value
                
                # Get veto decision
                decision = mechanism.assess_action(state, action, q_values)
                
                # Check decision has expected attributes
                attributes = ['vetoed', 'original_action', 'reason']
                for attr in attributes:
                    if not hasattr(decision, attr):
                        results['passed'] = False
                        results['failures'].append(f"Missing attribute in decision: {attr}")
                
                # Record decision for each action
                results['details'][f'action_{action}_vetoed'] = decision.vetoed
                results['details'][f'action_{action}_reason'] = decision.reason
        except Exception as e:
            results['passed'] = False
            results['failures'].append(f"Error processing actions: {str(e)}")
            
        # Test recording decisions
        try:
            mechanism.record_veto_decision(state, 0, True, 1, 0.5)
            results['details']['record_veto_success'] = True
        except Exception as e:
            results['passed'] = False
            results['failures'].append(f"Error recording veto decision: {str(e)}")
            results['details']['record_veto_success'] = False
            
        return results
    
    def test_consistency(self, mechanism):
        """
        Test consistency of veto decisions
        
        Args:
            mechanism: Veto mechanism to test
            
        Returns:
            Dictionary with test results
        """
        results = {
            'passed': True,
            'failures': [],
            'details': {}
        }
        
        # Create test states
        states = [
            self._create_test_state(health=0.8, ammo=0.8, shields=0.8),  # Good state
            self._create_test_state(health=0.2, ammo=0.2, shields=0.0)   # Bad state
        ]
        
        # Test consistency across multiple runs
        for i, state in enumerate(states):
            # Run multiple times and check consistency
            decisions_by_action = defaultdict(list)
            
            for run in range(5):
                for action in range(11):
                    q_values = np.zeros(11)
                    q_values[action] = 0.5
                    
                    decision = mechanism.assess_action(state, action, q_values)
                    decisions_by_action[action].append(decision.vetoed)
            
            # Check for consistency
            for action, decisions in decisions_by_action.items():
                # All decisions for same state+action should be the same
                consistent = all(d == decisions[0] for d in decisions)
                
                if not consistent:
                    results['passed'] = False
                    results['failures'].append(
                        f"Inconsistent decisions for state {i}, action {action}: {decisions}"
                    )
                    
                results['details'][f'state{i}_action{action}_consistent'] = consistent
                results['details'][f'state{i}_action{action}_vetoed'] = decisions[0]
        
        return results
    
    def test_edge_cases(self, mechanism):
        """
        Test veto mechanism with edge cases
        
        Args:
            mechanism: Veto mechanism to test
            
        Returns:
            Dictionary with test results
        """
        results = {
            'passed': True,
            'failures': [],
            'details': {}
        }
        
        # Edge cases to test
        test_cases = [
            {
                'name': 'critical_health',
                'state': self._create_test_state(health=0.01, ammo=0.5, shields=0.5),
                'expected_combat_veto': True
            },
            {
                'name': 'no_ammo',
                'state': self._create_test_state(health=0.5, ammo=0.01, shields=0.5),
                'expected_combat_veto': True
            },
            {
                'name': 'very_good_state',
                'state': self._create_test_state(health=1.0, ammo=1.0, shields=1.0),
                'expected_combat_veto': False
            },
            {
                'name': 'invalid_action',
                'state': self._create_test_state(health=0.5, ammo=0.5, shields=0.5),
                'action': 99,  # Invalid action index
                'expected_error_handled': True
            }
        ]
        
        # Run tests for each edge case
        for case in test_cases:
            state = case['state']
            
            if 'action' in case:
                # Specific action test
                action = case['action']
                q_values = np.zeros(11)
                q_values[min(action, 10)] = 0.5
                
                try:
                    decision = mechanism.assess_action(state, action, q_values)
                    results['details'][f"{case['name']}_error_handled"] = True
                except Exception as e:
                    results['details'][f"{case['name']}_error_handled"] = False
                    if case.get('expected_error_handled', False):
                        results['failures'].append(f"Failed to handle invalid action: {str(e)}")
                        results['passed'] = False
            else:
                # Test combat actions with this state
                combat_decisions = []
                for action in range(4, 8):  # Combat actions 4-7
                    q_values = np.zeros(11)
                    q_values[action] = 0.5
                    
                    decision = mechanism.assess_action(state, action, q_values)
                    combat_decisions.append(decision.vetoed)
                
                # Check if combat actions were vetoed as expected
                combat_vetoed = any(combat_decisions)
                results['details'][f"{case['name']}_combat_vetoed"] = combat_vetoed
                
                if combat_vetoed != case.get('expected_combat_veto', False):
                    results['passed'] = False
                    results['failures'].append(
                        f"Case {case['name']}: Expected combat veto {case['expected_combat_veto']}, "
                        f"got {combat_vetoed}"
                    )
        
        return results
    
    def test_performance(self, mechanism):
        """
        Test performance characteristics of veto mechanism
        
        Args:
            mechanism: Veto mechanism to test
            
        Returns:
            Dictionary with test results
        """
        results = {
            'passed': True,
            'failures': [],
            'details': {}
        }
        
        # Create test state with known properties
        state = self._create_test_state(health=0.5, ammo=0.5, shields=0.5)
        
        # Measure time to process 1000 actions
        start_time = time.time()
        iterations = 1000
        
        for _ in range(iterations):
            action = np.random.randint(0, 11)
            q_values = np.random.rand(11)
            decision = mechanism.assess_action(state, action, q_values)
            
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        results['details']['total_time'] = total_time
        results['details']['avg_time'] = avg_time
        results['details']['iterations'] = iterations
        
        # Check if performance is acceptable (< 1ms per decision)
        if avg_time > 0.001:
            results['failures'].append(
                f"Performance below threshold: {avg_time*1000:.2f}ms per decision"
            )
            # Don't fail the test because of performance - just warn
            # results['passed'] = False
        
        return results
    
    def test_agent_integration(self, mechanism, agent):
        """
        Test veto mechanism integration with agent
        
        Args:
            mechanism: Veto mechanism to test
            agent: Agent to use for testing
            
        Returns:
            Dictionary with test results
        """
        results = {
            'passed': True,
            'failures': [],
            'details': {}
        }
        
        # Create environment for testing
        env = self.environment_class()
        state = env.reset()
        
        # Record veto decisions
        veto_decisions = []
        rewards = []
        
        # Run for several steps
        for _ in range(20):
            # Get action from agent
            action, q_values = agent.select_action(state)
            
            # Check veto
            decision = mechanism.assess_action(state, action, q_values)
            veto_decisions.append(decision.vetoed)
            
            # Take action (vetoed or not)
            if decision.vetoed:
                # Use agent's safe action selection
                if hasattr(agent, 'select_safe_action'):
                    action = agent.select_safe_action(state, action)
                else:
                    # Default to random alternative
                    action = (action + 1) % 11
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            
            # Store transition for agent
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            if done:
                break
        
        # Calculate statistics
        results['details']['veto_count'] = sum(veto_decisions)
        results['details']['veto_rate'] = sum(veto_decisions) / len(veto_decisions)
        results['details']['total_reward'] = sum(rewards)
        results['details']['avg_reward'] = sum(rewards) / len(rewards)
        
        # Test pass/fail criteria based on reasonable veto rate
        if results['details']['veto_rate'] < 0.05 or results['details']['veto_rate'] > 0.8:
            results['passed'] = False
            results['failures'].append(
                f"Veto rate {results['details']['veto_rate']:.2f} is outside reasonable range [0.05, 0.8]"
            )
        
        return results
    
    def validate_against_ground_truth(self, mechanism):
        """
        Validate veto mechanism against ground truth
        
        Args:
            mechanism: Veto mechanism to test
            
        Returns:
            Dictionary with test results
        """
        results = {
            'passed': True,
            'failures': [],
            'details': {}
        }
        
        # Create test states
        states = [
            self._create_test_state(health=0.8, ammo=0.8, shields=0.8),  # Good state
            self._create_test_state(health=0.5, ammo=0.5, shields=0.5),  # Medium state
            self._create_test_state(health=0.2, ammo=0.2, shields=0.0)   # Bad state
        ]
        
        # Create environment for more realistic states
        env = self.environment_class()
        for _ in range(5):
            states.append(GameState(raw_state=env.reset()))
        
        # Validate each state with various actions
        correct_count = 0
        total_count = 0
        results_by_state = []
        
        for i, state in enumerate(states):
            state_results = {'state_idx': i, 'actions': []}
            
            for action in range(11):
                q_values = np.random.rand(11)
                q_values[action] = np.random.rand()  # Randomize Q-value
                
                # Get ground truth
                gt_vetoed, gt_confidence, gt_explanation = self.oracle.should_veto(
                    state, action, q_values
                )
                
                # Get mechanism decision
                decision = mechanism.assess_action(state, action, q_values)
                
                # Check correctness
                correct = decision.vetoed == gt_vetoed
                if correct:
                    correct_count += 1
                
                total_count += 1
                
                # Record results for this action
                state_results['actions'].append({
                    'action': action,
                    'mechanism_vetoed': decision.vetoed,
                    'ground_truth_vetoed': gt_vetoed,
                    'correct': correct,
                    'ground_truth_confidence': gt_confidence
                })
            
            results_by_state.append(state_results)
        
        # Calculate accuracy
        accuracy = correct_count / total_count if total_count > 0 else 0
        results['details']['accuracy'] = accuracy
        results['details']['correct_count'] = correct_count
        results['details']['total_count'] = total_count
        results['details']['results_by_state'] = results_by_state
        
        # Test pass/fail criteria based on accuracy
        if accuracy < 0.6:
            results['passed'] = False
            results['failures'].append(
                f"Accuracy {accuracy:.2f} is below threshold 0.6"
            )
            
        return results
    
    def compare_mechanisms(self, mechanisms, agent=None):
        """
        Compare multiple veto mechanisms
        
        Args:
            mechanisms: List of veto mechanisms to compare
            agent: Optional agent to use for testing
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        # Run tests for each mechanism
        for mechanism in mechanisms:
            mechanism_name = mechanism.__class__.__name__
            print(f"\nTesting {mechanism_name}...")
            
            # Run all tests
            results = self.run_all_tests(mechanism, agent)
            
            # Store overall score for comparison
            comparison[mechanism_name] = {
                'score': results['overall_score'],
                'ground_truth_accuracy': results['ground_truth'].get('details', {}).get('accuracy', 0),
                'passed_all_tests': all(r.get('passed', False) for r in results.values() 
                                     if isinstance(r, dict) and 'passed' in r)
            }
        
        # Calculate rankings
        mechanisms_ranked = sorted(
            comparison.keys(),
            key=lambda m: comparison[m]['score'],
            reverse=True
        )
        
        # Save comparison results
        if self.save_dir:
            comparison_path = os.path.join(self.save_dir, 'mechanisms_comparison.json')
            with open(comparison_path, 'w') as f:
                json.dump({
                    'comparison': comparison,
                    'ranking': mechanisms_ranked
                }, f, indent=2)
        
        return {
            'comparison': comparison,
            'ranking': mechanisms_ranked
        }
    
    def _create_test_state(self, health=0.5, ammo=0.5, shields=0.5):
        """Create test state with specified properties"""
        # Create simple grid
        grid_size = 50
        grid_data = np.ones(grid_size * grid_size)
        
        # Create agent stats
        agent_stats = {
            'health': health,
            'ammo': ammo,
            'shields': shields
        }
        
        # Create GameState
        return GameState(grid_data, agent_stats)
    
    def _calculate_overall_score(self, test_results):
        """Calculate overall score from test results"""
        score = 0.0
        
        # Basic functionality (2 points)
        if test_results.get('basic', {}).get('passed', False):
            score += 2.0
            
        # Consistency (2 points)
        if test_results.get('consistency', {}).get('passed', False):
            score += 2.0
            
        # Edge cases (1.5 points)
        if test_results.get('edge_cases', {}).get('passed', False):
            score += 1.5
            
        # Performance (0.5 points)
        if test_results.get('performance', {}).get('passed', False):
            score += 0.5
            
        # Agent integration (1 point)
        if test_results.get('agent_integration', {}).get('passed', False):
            score += 1.0
            
        # Ground truth validation (3 points - weighted by accuracy)
        ground_truth = test_results.get('ground_truth', {})
        if ground_truth.get('passed', False):
            score += 3.0
        else:
            # Partial credit based on accuracy
            accuracy = ground_truth.get('details', {}).get('accuracy', 0)
            score += 3.0 * (accuracy / 0.6)  # Scale by threshold
            
        return min(10.0, score)  # Cap at 10 points
    
    def _save_results(self, mechanism_name, test_results):
        """Save test results to file"""
        if not self.save_dir:
            return
            
        # Create mechanism directory
        mechanism_dir = os.path.join(self.save_dir, mechanism_name)
        os.makedirs(mechanism_dir, exist_ok=True)
        
        # Save results to JSON
        results_path = os.path.join(mechanism_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            # Convert values to serializable types
            serializable_results = json.loads(
                json.dumps(test_results, default=lambda o: str(o))
            )
            json.dump(serializable_results, f, indent=2)
            
        print(f"Test results saved to {results_path}")


class TestCase:
    """
    Predefined test case for veto mechanism testing.
    """
    def __init__(self, name, state, action, q_values=None, 
                expected_veto=None, description=""):
        self.name = name
        self.state = state
        self.action = action
        self.q_values = q_values
        self.expected_veto = expected_veto
        self.description = description

class TestEnvironment:
    """
    Simple environment for testing purposes.
    """
    def __init__(self, grid_size=50):
        self.grid_size = grid_size
        self.action_space_n = 11
        self.reset()
        
    def reset(self):
        """Reset environment"""
        # Create simple state
        self.agent_health = 100
        self.agent_ammo = 15
        self.agent_shields = 3
        self.agent_pos = (self.grid_size // 2, self.grid_size // 2)
        
        return self._get_state()
        
    def step(self, action):
        """Take step with action"""
        # Simple implementation for testing
        reward = 0
        done = False
        
        # Movement
        if action <= 3:
            reward = 0.1
            
        # Combat
        elif action <= 7:
            if self.agent_ammo > 0:
                self.agent_ammo -= 1
                reward = 0.5
            else:
                reward = -0.5
                
        # Special actions
        else:
            if action == 9:  # Use cover
                reward = 0.2
            else:
                reward = 0.0
                
        # Small random health change
        self.agent_health = max(0, min(100, self.agent_health + np.random.randint(-5, 5)))
        
        # Check if done
        if self.agent_health <= 0:
            done = True
            reward = -1.0
            
        return self._get_state(), reward, done, {}
        
    def _get_state(self):
        """Get current state"""
        # Create simple state for testing
        grid = np.ones(self.grid_size * self.grid_size)
        
        # Add agent stats
        agent_stats = np.array([
            self.agent_health / 100.0,
            self.agent_ammo / 30.0,
            self.agent_shields / 5.0
        ])
        
        return np.concatenate([grid, agent_stats])


def run_standard_veto_tests(veto_mechanisms, save_dir=None):
    """
    Run standard tests for veto mechanisms
    
    Args:
        veto_mechanisms: Dictionary mapping names to veto mechanisms
        save_dir: Directory to save results
        
    Returns:
        Test results and comparison
    """
    # Create test environment
    env_class = TestEnvironment
    
    # Create test suite
    test_suite = VetoTestSuite(env_class, save_dir)
    
    # Run tests for each mechanism
    results = {}
    for name, mechanism in veto_mechanisms.items():
        print(f"\nTesting {name}...")
        results[name] = test_suite.run_all_tests(mechanism)
        
    # Compare mechanisms
    comparison = test_suite.compare_mechanisms(list(veto_mechanisms.values()))
    
    return results, comparison