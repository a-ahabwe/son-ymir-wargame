import unittest
import numpy as np
import torch
import os
import sys
from pathlib import Path
import random

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ai.models import DuelingDQN
from src.ai.agent import RLAgent
from src.ai.uncertainty import UncertaintyEstimator
from src.game.environment import GameEnvironment

class RLBehaviorTest(unittest.TestCase):
    """Test the behavior of reinforcement learning components."""
    
    def setUp(self):
        """Set up test environment with necessary components."""
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        
        # Create small environment and agent for testing
        self.env = GameEnvironment(grid_size=20, seed=42)
        self.state_size = self.env.state_size
        self.action_size = self.env.action_space_n
        
        # Create agent
        self.agent = RLAgent(self.state_size, self.action_size)
        
        # Create uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(self.agent.model)
        
        # Create a deterministic state for testing
        self.test_state = self.env.reset()
        
    def test_agent_initialization(self):
        """Test that agent initialization creates valid models."""
        self.assertIsInstance(self.agent.model, DuelingDQN)
        self.assertIsInstance(self.agent.target_model, DuelingDQN)
        
        # Check model dimensions
        self.assertEqual(self.agent.model.state_size, self.state_size)
        self.assertEqual(self.agent.model.action_size, self.action_size)
        
        # Check target model matches main model initially
        for p1, p2 in zip(self.agent.model.parameters(), self.agent.target_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))
    
    def test_action_selection(self):
        """Test that agent can select actions properly."""
        # Test basic action selection
        action, q_values = self.agent.select_action(self.test_state, explore=False)
        
        # Action should be a valid index
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_size)
        
        # Q-values should match the action size
        self.assertEqual(len(q_values), self.action_size)
        
        # Test with exploration
        actions = []
        for _ in range(20):
            action, _ = self.agent.select_action(self.test_state, explore=True)
            actions.append(action)
            
        # With exploration enabled, should see some variety in actions
        self.assertGreater(len(set(actions)), 1)
    
    def test_experience_replay(self):
        """Test experience replay buffer functionality."""
        # Start with empty buffer
        self.agent.replay_buffer.clear()
        
        # Store some transitions
        for _ in range(5):
            state = np.random.rand(self.state_size)
            action = random.randint(0, self.action_size - 1)
            reward = random.random()
            next_state = np.random.rand(self.state_size)
            done = random.choice([True, False])
            
            self.agent.store_transition(state, action, reward, next_state, done)
            
        # Check buffer size
        self.assertEqual(len(self.agent.replay_buffer), 5)
    
    def test_training_step(self):
        """Test that agent can perform a training step."""
        # Prepare buffer with sufficient samples
        buffer_size = max(self.agent.batch_size + 10, 50)
        
        # Clear existing buffer
        self.agent.replay_buffer.clear()
        
        # Fill buffer
        for _ in range(buffer_size):
            state = np.random.rand(self.state_size)
            action = random.randint(0, self.action_size - 1)
            reward = random.random() * 2 - 1  # Range [-1, 1]
            next_state = np.random.rand(self.state_size)
            done = random.choice([True, False])
            
            self.agent.store_transition(state, action, reward, next_state, done)
        
        # Get initial model parameters
        initial_params = [p.clone().detach() for p in self.agent.model.parameters()]
        
        # Perform training step
        loss = self.agent.train()
        
        # Check loss is a valid number
        self.assertIsNotNone(loss)
        self.assertFalse(np.isnan(loss))
        
        # Verify parameters changed after training
        updated = False
        for p1, p2 in zip(initial_params, self.agent.model.parameters()):
            if not torch.equal(p1, p2):
                updated = True
                break
                
        self.assertTrue(updated, "Model parameters did not change after training")
    
    def test_target_update(self):
        """Test that target network gets updated correctly."""
        # First make sure target and main networks match
        self.agent.update_target_model()
        
        # Train the main network to create a difference
        self.test_training_step()  # This will update the main network
        
        # Check that main and target networks now differ
        main_params = list(self.agent.model.parameters())
        target_params = list(self.agent.target_model.parameters())
        
        # Find at least one parameter that differs
        param_differs = False
        for p1, p2 in zip(main_params, target_params):
            if not torch.equal(p1, p2):
                param_differs = True
                break
                
        self.assertTrue(param_differs, "Main and target networks should differ after training")
        
        # Now update target network again
        self.agent.update_target_model()
        
        # Verify they match again
        for p1, p2 in zip(self.agent.model.parameters(), self.agent.target_model.parameters()):
            self.assertTrue(torch.equal(p1, p2), "Parameters don't match after target update")
    
    def test_save_load_model(self):
        """Test that agent can save and load models."""
        # Create a temporary file path
        temp_path = 'temp_model_test.pt'
        
        try:
            # Save model
            self.agent.save(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Train to modify weights
            self.test_training_step()
            
            # Get current weights
            original_weights = [p.clone().detach() for p in self.agent.model.parameters()]
            
            # Create a new agent
            new_agent = RLAgent(self.state_size, self.action_size)
            
            # Verify weights differ
            weights_differ = False
            for p1, p2 in zip(original_weights, new_agent.model.parameters()):
                if not torch.allclose(p1, p2):
                    weights_differ = True
                    break
                    
            self.assertTrue(weights_differ, "New agent should have different weights")
            
            # Load saved weights into new agent
            new_agent.load(temp_path)
            
            # Verify weights now match
            for p1, p2 in zip(original_weights, new_agent.model.parameters()):
                self.assertTrue(torch.allclose(p1, p2), "Weights don't match after loading")
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_safe_action_selection(self):
        """Test safe action selection functionality."""
        # Get a normal action
        action, q_values = self.agent.select_action(self.test_state)
        
        # Select a safe alternative
        safe_action = self.agent.select_safe_action(self.test_state, action)
        
        # Should be different from original action
        self.assertNotEqual(safe_action, action)
        
        # Should be a valid action
        self.assertGreaterEqual(safe_action, 0)
        self.assertLess(safe_action, self.action_size)
    
    def test_agent_behavior_in_environment(self):
        """Test agent behavior in the actual environment."""
        # Reset environment
        state = self.env.reset()
        
        # Track metrics
        total_reward = 0
        steps = 0
        
        # Run for a few steps
        for _ in range(20):
            # Select action
            action, _ = self.agent.select_action(state)
            
            # Take action
            next_state, reward, done, _ = self.env.step(action)
            
            # Store transition (helps verify this doesn't crash)
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # Update metrics
            total_reward += reward
            steps += 1
            
            # Update state
            state = next_state
            
            if done:
                break
        
        # Verify we could complete steps without errors
        self.assertGreater(steps, 0)
        
    def test_uncertainty_estimation(self):
        """Test uncertainty estimation behavior."""
        # Get uncertainty for a state
        uncertainty = self.uncertainty_estimator.estimate_uncertainty(self.test_state)
        
        # Check dimensions
        self.assertEqual(uncertainty.shape[1], self.action_size)
        
        # Values should be in [0, 1] range
        self.assertTrue(np.all(uncertainty >= 0))
        self.assertTrue(np.all(uncertainty <= 1))
        
        # Get uncertainty for specific action
        action_uncertainty = self.uncertainty_estimator.get_action_uncertainty(
            self.test_state, 0
        )
        
        # Should be a scalar
        self.assertTrue(np.isscalar(action_uncertainty))
        
        # Value should be in [0, 1] range
        self.assertGreaterEqual(action_uncertainty, 0)
        self.assertLessEqual(action_uncertainty, 1)
    
    def test_training_effects_on_q_values(self):
        """Test that training affects Q-values in a meaningful way."""
        # Clear replay buffer
        self.agent.replay_buffer.clear()
        
        # Generate specific training data that should lead to predictable changes
        # Create transitions where action 0 consistently gives positive reward
        for _ in range(100):
            state = np.random.rand(self.state_size)
            next_state = np.random.rand(self.state_size)
            
            # Action 0 always gives positive reward
            self.agent.store_transition(state, 0, 1.0, next_state, False)
            
            # Action 1 always gives negative reward
            self.agent.store_transition(state, 1, -1.0, next_state, False)
        
        # Get initial Q-values for a test state
        initial_q = self.agent.get_q_values(self.test_state)
        
        # Train for several steps
        for _ in range(10):
            self.agent.train()
            
        # Get updated Q-values
        updated_q = self.agent.get_q_values(self.test_state)
        
        # The Q-value for action 0 should increase relative to action 1
        q0_diff = updated_q[0] - initial_q[0]
        q1_diff = updated_q[1] - initial_q[1]
        
        # Action 0 should improve more than action 1
        self.assertGreater(q0_diff, q1_diff)
        
    def test_deterministic_behavior(self):
        """Test that with fixed seeds, behavior is deterministic."""
        # Reset seeds
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        
        # First run
        actions1 = []
        state = self.test_state
        for _ in range(10):
            action, _ = self.agent.select_action(state, explore=False)
            actions1.append(action)
        
        # Reset seeds
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        
        # Second run - should match first run
        actions2 = []
        state = self.test_state
        for _ in range(10):
            action, _ = self.agent.select_action(state, explore=False)
            actions2.append(action)
        
        # Actions should be identical
        self.assertEqual(actions1, actions2)

if __name__ == '__main__':
    unittest.main()