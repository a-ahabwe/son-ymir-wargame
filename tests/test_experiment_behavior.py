import unittest
import os
import shutil
import tempfile
import sys
import json
import numpy as np
from pathlib import Path
import itertools

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.experiment.improved_experiment import ImprovedExperiment
from src.experiment.scenarios import ScenarioGenerator
from src.game.environment import GameEnvironment

class ExperimentBehaviorTest(unittest.TestCase):
    """Test the behavior of the experiment system."""
    
    def setUp(self):
        """Set up test environment for experiments."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a small experiment for testing
        self.experiment = ImprovedExperiment(
            output_dir=self.temp_dir,
            seed=42
        )
        
        # Add some test participants
        self.experiment.setup_participants(num_participants=2, virtual=True)
        
        # Add simple conditions
        self.experiment.setup_conditions(["threshold"], include_control=True)
        
        # Add test scenarios
        self.experiment.setup_scenarios(num_scenarios=2)
    
    def tearDown(self):
        """Clean up temporary files."""
        # Remove the temp directory
        shutil.rmtree(self.temp_dir)
    
    def test_experiment_initialization(self):
        """Test experiment initialization creates valid structures."""
        # Check basic properties
        self.assertIsNotNone(self.experiment.experiment_id)
        self.assertEqual(self.experiment.seed, 42)
        
        # Check config
        self.assertEqual(len(self.experiment.config["participants"]), 2)
        self.assertEqual(len(self.experiment.config["conditions"]), 2)  # threshold + control
        self.assertEqual(len(self.experiment.config["scenarios"]), 2)
        
        # Check output directory was created
        output_dir = self.experiment.output_dir
        self.assertTrue(os.path.exists(output_dir))
    
    def test_hypothesis_creation(self):
        """Test adding and tracking research hypotheses."""
        # Add a hypothesis
        hypothesis_id = self.experiment.add_hypothesis(
            "Treatment condition improves performance compared to control",
            null_hypothesis="No difference between treatment and control",
            type="superiority"
        )
        
        # Check it was stored
        self.assertGreater(len(self.experiment.config["hypotheses"]), 0)
        
        # Check hypothesis properties
        hypothesis = self.experiment.config["hypotheses"][0]
        self.assertEqual(hypothesis["id"], hypothesis_id)
        self.assertEqual(hypothesis["type"], "superiority")
        self.assertIsNone(hypothesis["result"])  # No result yet
    
    def test_scenario_generation(self):
        """Test scenario generation produces valid test scenarios."""
        # Create new scenarios
        scenarios = self.experiment.setup_scenarios(num_scenarios=3)
        
        # Check basic properties
        self.assertEqual(len(scenarios), 3)
        
        # Check difficulty progression
        difficulties = [s.difficulty for s in scenarios]
        self.assertTrue(all(0 <= d <= 1 for d in difficulties))
        
        # Should be in ascending difficulty
        self.assertEqual(difficulties, sorted(difficulties))
    
    def test_counterbalancing(self):
        """Test that experiment properly counterbalances conditions."""
        # Create a new experiment with more participants
        experiment = ImprovedExperiment(
            output_dir=self.temp_dir,
            seed=42
        )
        
        # Add more conditions and participants for proper counterbalancing
        experiment.setup_conditions(["A", "B", "C"])
        experiment.setup_participants(num_participants=6, balanced=True)
        
        # Check that each condition order is different
        condition_orders = [p["condition_order"] for p in experiment.participants]
        
        # With 3 conditions and 6 participants, we should see all possible orders
        # Number of permutations = 3! = 6
        unique_orders = set(tuple(order) for order in condition_orders)
        self.assertEqual(len(unique_orders), 6)
        
    def test_experiment_config_saving(self):
        """Test that experiment configuration is properly saved."""
        # Save the configuration
        self.experiment._save_config()
        
        # Check file exists
        config_path = os.path.join(self.experiment.output_dir, "experiment_config.json")
        self.assertTrue(os.path.exists(config_path))
        
        # Load it and verify content
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Check key components are present
        self.assertEqual(config["experiment_id"], self.experiment.experiment_id)
        self.assertEqual(config["seed"], 42)
        self.assertEqual(len(config["participants"]), 2)
        self.assertEqual(len(config["conditions"]), 2)
        
    def test_power_analysis(self):
        """Test power analysis calculations."""
        # Create new experiment and run power analysis
        experiment = ImprovedExperiment(
            output_dir=self.temp_dir,
            seed=42
        )
        
        # Run with a small sample size
        experiment.setup_participants(num_participants=10)
        
        # Check power analysis was performed
        self.assertIn("power_analysis", experiment.config)
        
        # Check for key metrics
        power_analysis = experiment.config["power_analysis"]
        self.assertIn("estimated_power", power_analysis)
        self.assertIn("recommended_sample_size", power_analysis)
        
        # Small sample should have limited power
        self.assertLess(power_analysis["estimated_power"], 0.8)
        
        # Recommended sample should be larger
        self.assertGreater(power_analysis["recommended_sample_size"], 10)
        
        # Run with larger sample
        experiment.setup_participants(num_participants=100)
        
        # Power should be higher
        self.assertGreater(experiment.config["power_analysis"]["estimated_power"], 
                           power_analysis["estimated_power"])
        
    def test_veto_simulation(self):
        """Test the human veto decision simulation."""
        # Create a realistic state, action and veto info
        state = np.zeros(10)
        state[-3:] = [0.2, 0.1, 0.0]  # Low resources
        action = 2
        q_values = np.array([0.2, 0.1, -0.1, 0.5])
        veto_info = {
            "is_risky": True,
            "risk_level": 0.8,
            "uncertainty": 0.7
        }
        
        # Test decision for different participant profiles
        test_participant = {"veto_behavior": "conservative"}
        conservative_decision = self.experiment.simulate_veto_decision(
            test_participant, state, action, q_values, veto_info
        )
        
        # Conservative participant should veto high-risk actions
        self.assertTrue(conservative_decision)
        
        # Test permissive participant
        test_participant = {"veto_behavior": "permissive"}
        permissive_decision = self.experiment.simulate_veto_decision(
            test_participant, state, action, q_values, veto_info
        )
        
        # Results might vary due to randomness, but we don't strictly check the value
        # Just ensure it returns a valid boolean
        self.assertIsInstance(permissive_decision, bool)
    
    def test_running_experiment(self):
        """Test running a small experiment."""
        # Create minimal experiment
        experiment = ImprovedExperiment(
            output_dir=self.temp_dir,
            seed=42
        )
        
        # Configure minimal settings
        experiment.setup_participants(num_participants=1, virtual=True)
        experiment.setup_conditions(["threshold"], include_control=False)
        experiment.setup_scenarios(num_scenarios=1)
        
        # Run abbreviated experiment (mock components to speedup test)
        # Override _run_scenario to do minimal work
        # original_run_scenario = experiment._run_scenario # Removed mocking
        
        # def mock_run_scenario(*args, **kwargs):
        #     return {"total_reward": 10, "total_steps": 5, "veto_count": 2, "successful_vetos": 1}
            
        # experiment._run_scenario = mock_run_scenario # Removed mocking
        
        try:
            # Run experiment
            output_dir = experiment.run_experiment()
            
            # Check that experiment completed
            self.assertTrue(os.path.exists(output_dir))
            
            # Check for expected output files
            config_path = os.path.join(output_dir, "experiment_config.json")
            self.assertTrue(os.path.exists(config_path))
            
            # Should have generated session data
            session_files = [f for f in os.listdir(output_dir) if f.startswith('p')]
            self.assertGreater(len(session_files), 0)
        finally:
            # Restore original method
            # experiment._run_scenario = original_run_scenario # No longer needed
            pass # No cleanup needed if not mocking
    
    def test_analysis_functionality(self):
        """Test experiment analysis functionality."""
        # Create a mock experiment with fake session data
        experiment = ImprovedExperiment(
            output_dir=self.temp_dir,
            seed=42
        )
        
        # Create minimal experiment components
        experiment.setup_participants(num_participants=1, virtual=True)
        experiment.setup_conditions(["threshold", "uncertainty"])
        
        # Run abbreviated experiment with mock data
        original_run_experiment = experiment.run_experiment
        
        def mock_run_experiment():
            # Create fake session files
            self._create_mock_session_data(experiment.output_dir)
            return experiment.output_dir
            
        experiment.run_experiment = mock_run_experiment
        
        try:
            # Run mock experiment
            experiment.run_experiment()
            
            # Run analysis
            analysis_results = experiment.analyze_results()
            
            # Check for expected analysis outputs
            self.assertIn("condition_metrics", analysis_results)
            
            # Check analysis directory was created
            analysis_dir = os.path.join(experiment.output_dir, "analysis")
            self.assertTrue(os.path.exists(analysis_dir))
            
            # Should have generated visualizations
            visualization_files = [f for f in os.listdir(analysis_dir) if f.endswith('.png')]
            self.assertGreater(len(visualization_files), 0)
        finally:
            # Restore original method
            experiment.run_experiment = original_run_experiment
    
    def _create_mock_session_data(self, output_dir):
        """Create mock session data files for testing analysis."""
        # Create test data
        session_data = {
            "participant_id": 1,
            "condition": "threshold",
            "start_time": 1600000000,
            "end_time": 1600001000,
            "duration": 1000,
            "actions": [
                {"timestamp": 1600000010, "action": 0, "reward": 0.5},
                {"timestamp": 1600000020, "action": 1, "reward": -0.2},
                {"timestamp": 1600000030, "action": 2, "reward": 1.0}
            ],
            "veto_decisions": [
                {"timestamp": 1600000050, "original_action": 3, "vetoed": True, "reward": 0.8},
                {"timestamp": 1600000060, "original_action": 4, "vetoed": False, "reward": -0.5}
            ]
        }
        
        # Create another session with different condition
        session_data2 = session_data.copy()
        session_data2["condition"] = "uncertainty"
        
        # Save to files
        with open(os.path.join(output_dir, "p1_threshold_1600000000.json"), 'w') as f:
            json.dump(session_data, f)
            
        with open(os.path.join(output_dir, "p1_uncertainty_1600002000.json"), 'w') as f:
            json.dump(session_data2, f)
    
    def test_environment_integration(self):
        """Test integration with the game environment."""
        # Create environment
        env = GameEnvironment(grid_size=10, seed=42)
        
        # Create a scenario generator and scenarios
        generator = ScenarioGenerator(seed=42)
        scenarios = generator.generate_scenarios(2)
        
        # Apply a scenario to the environment
        env = scenarios[0].apply_to_environment(env)
        
        # Verify environment was updated
        self.assertEqual(env.grid_size, scenarios[0].params['grid_size'])
        
        # Test running the scenario through the experiment
        # Create a veto mechanism
        veto_mechanism = self.experiment._create_veto_mechanism("threshold")
        
        # Test participant
        participant = {"id": 1, "virtual": True, "veto_behavior": "moderate"}
        
        # Run scenario
        self.experiment._setup_components()
        result = self.experiment._run_scenario(
            participant, env, veto_mechanism, scenarios[0], max_steps=5
        )
        
        # Check result contains expected fields
        self.assertIn("total_reward", result)
        self.assertIn("total_steps", result)
        self.assertIn("veto_count", result)
    
    def test_experiment_reproducibility(self):
        """Test that experiments are reproducible with fixed seed."""
        # Create two identical experiments
        exp1 = ImprovedExperiment(seed=42)
        exp2 = ImprovedExperiment(seed=42)
        
        # Create identical scenarios
        scenarios1 = exp1.setup_scenarios(num_scenarios=3)
        scenarios2 = exp2.setup_scenarios(num_scenarios=3)
        
        # Parameters should match
        for s1, s2 in zip(scenarios1, scenarios2):
            self.assertEqual(s1.difficulty, s2.difficulty)
            for k in s1.params:
                self.assertEqual(s1.params[k], s2.params[k])
        
        # Create identical participants
        exp1.setup_participants(num_participants=3, virtual=True)
        exp2.setup_participants(num_participants=3, virtual=True)
        
        # Participants should have same properties
        for p1, p2 in zip(exp1.participants, exp2.participants):
            self.assertEqual(p1["id"], p2["id"])
            self.assertEqual(p1["veto_behavior"], p2["veto_behavior"])
            self.assertEqual(p1["condition_order"], p2["condition_order"])

if __name__ == '__main__':
    unittest.main()