import unittest
import torch
import numpy as np
from src.ai.models import DuelingDQN, EnsembleDQN
from src.ai.agent import RLAgent
from src.ai.uncertainty import UncertaintyEstimator

class TestDuelingDQN(unittest.TestCase):
    def setUp(self):
        self.state_size = 100
        self.action_size = 11
        self.model = DuelingDQN(self.state_size, self.action_size)
        
    def test_initialization(self):
        # Check that model layers exist
        self.assertIsNotNone(self.model.features)
        self.assertIsNotNone(self.model.value_stream)
        self.assertIsNotNone(self.model.advantage_stream)
        
    def test_forward_numpy(self):
        # Test with numpy input
        state = np.random.rand(self.state_size)
        output = self.model(state)
        
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, self.action_size))
        
    def test_forward_batch(self):
        # Test with batch input
        batch_size = 10
        state_batch = torch.rand(batch_size, self.state_size)
        output = self.model(state_batch)
        
        self.assertEqual(output.shape, (batch_size, self.action_size))
        
class TestEnsembleDQN(unittest.TestCase):
    def setUp(self):
        self.state_size = 100
        self.action_size = 11
        self.ensemble_size = 3
        self.ensemble = EnsembleDQN(self.state_size, self.action_size, self.ensemble_size)
        
    def test_initialization(self):
        self.assertEqual(len(self.ensemble.models), self.ensemble_size)
        
    def test_predict(self):
        state = np.random.rand(self.state_size)
        prediction = self.ensemble.predict(state)
        
        self.assertEqual(prediction.shape, (1, self.action_size))
        
    def test_uncertainty(self):
        state = np.random.rand(self.state_size)
        uncertainty = self.ensemble.uncertainty(state)
        
        self.assertEqual(uncertainty.shape, (1, self.action_size))
        
class TestRLAgent(unittest.TestCase):
    def setUp(self):
        self.state_size = 100
        self.action_size = 11
        self.agent = RLAgent(self.state_size, self.action_size)
        
    def test_initialization(self):
        self.assertIsInstance(self.agent.model, DuelingDQN)
        self.assertIsInstance(self.agent.target_model, DuelingDQN)
        
    def test_select_action(self):
        state = np.random.rand(self.state_size)
        action, q_values = self.agent.select_action(state)
        
        self.assertTrue(0 <= action < self.action_size)
        self.assertEqual(len(q_values), self.action_size)
        
    def test_store_transition(self):
        state = np.random.rand(self.state_size)
        next_state = np.random.rand(self.state_size)
        action = 0
        reward = 1.0
        done = False
        
        initial_buffer_len = len(self.agent.replay_buffer)
        self.agent.store_transition(state, action, reward, next_state, done)
        
        self.assertEqual(len(self.agent.replay_buffer), initial_buffer_len + 1)
        
class TestUncertaintyEstimator(unittest.TestCase):
    def setUp(self):
        self.state_size = 100
        self.action_size = 11
        
        # Create model
        self.model = DuelingDQN(self.state_size, self.action_size)
        
        # Create uncertainty estimator
        self.estimator = UncertaintyEstimator(self.model)
        
    def test_epistemic_uncertainty(self):
        state = np.random.rand(self.state_size)
        uncertainty = self.estimator.epistemic_uncertainty(state)
        
        self.assertEqual(len(uncertainty), self.action_size)
        
    def test_aleatoric_uncertainty(self):
        state = np.random.rand(self.state_size)
        uncertainty = self.estimator.aleatoric_uncertainty(state)
        
        self.assertEqual(len(uncertainty), self.action_size)
        
    def test_decision_uncertainty(self):
        state = np.random.rand(self.state_size)
        action = 0
        
        uncertainty = self.estimator.decision_uncertainty(state, action)
        
        self.assertIsInstance(uncertainty, (float, np.float64, np.float32))

if __name__ == '__main__':
    unittest.main()