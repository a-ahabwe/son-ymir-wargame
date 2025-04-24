import unittest
import numpy as np
import time
from src.veto.risk_assessment import RiskAssessor
from src.veto.veto_mechanism import VetoMechanism, ThresholdVetoMechanism, UncertaintyVetoMechanism
from src.ai.uncertainty import UncertaintyEstimator
from src.ai.models import DuelingDQN

class TestRiskAssessor(unittest.TestCase):
    def setUp(self):
        self.risk_assessor = RiskAssessor()
        
    def test_extract_features(self):
        # Create dummy state and action
        state = np.zeros(103)  # 100 grid cells + 3 agent stats
        state[-3:] = [0.5, 0.6, 0.7]  # health, ammo, shields
        action = 0
        q_values = np.random.rand(11)
        
        features = self.risk_assessor.extract_features(state, action, q_values)
        
        # Check feature dimensions
        self.assertEqual(len(features), 17)  # 3 agent stats + 11 one-hot action + 3 q-value features
        
    def test_is_high_risk(self):
        # Create dummy state and action
        state = np.zeros(103)
        state[-3:] = [0.2, 0.1, 0]  # Low health, low ammo, no shields
        action = 4  # Shoot
        
        # Should be high risk due to low ammo for shooting
        is_risky, reason = self.risk_assessor.is_high_risk(state, action)
        
        self.assertTrue(is_risky)
        self.assertIn("ammo", reason.lower())
        
        # Try a safer action
        action = 9  # Use cover
        is_risky, reason = self.risk_assessor.is_high_risk(state, action)
        
        # Should not be high risk (using cover with low health is good)
        self.assertFalse(is_risky)
        
class TestVetoMechanism(unittest.TestCase):
    def setUp(self):
        self.veto = VetoMechanism(timeout=5)
        
    def test_initialization(self):
        self.assertEqual(self.veto.timeout, 5)
        self.assertIsInstance(self.veto.risk_assessor, RiskAssessor)
        self.assertEqual(len(self.veto.veto_history), 0)
        
    def test_record_veto_decision(self):
        state = np.zeros(10)
        action = 0
        
        self.veto.record_veto_decision(state, action, vetoed=True)
        
        self.assertEqual(len(self.veto.veto_history), 1)
        entry = self.veto.veto_history[0]
        self.assertEqual(entry['action'], action)
        self.assertTrue(entry['vetoed'])
        
class TestThresholdVetoMechanism(unittest.TestCase):
    def setUp(self):
        self.veto = ThresholdVetoMechanism(threshold=0.7, timeout=5)
        
    def test_assess_action(self):
        # Create a state with low health and ammo
        state = np.zeros(103)
        state[-3:] = [0.2, 0.1, 0]  # Low health, low ammo, no shields
        
        # Test with a high-risk action (shooting with low ammo)
        action = 4  # Shoot
        need_veto, reasoning, risk_reason = self.veto.assess_action(state, action)
        
        self.assertTrue(need_veto)
        self.assertIsNotNone(reasoning)
        self.assertIsNotNone(risk_reason)
        
class TestUncertaintyVetoMechanism(unittest.TestCase):
    def setUp(self):
        # Create model and uncertainty estimator
        state_size = 103
        action_size = 11
        model = DuelingDQN(state_size, action_size)
        uncertainty_estimator = UncertaintyEstimator(model)
        
        self.veto = UncertaintyVetoMechanism(uncertainty_estimator, uncertainty_threshold=0.5, timeout=5)
        
    def test_assess_action_with_uncertainty(self):
        # Create dummy state
        state = np.zeros(103)
        state[-3:] = [0.5, 0.5, 0.5]  # Normal health, ammo, shields
        action = 0  # Move up
        
        # This will likely trigger veto due to the random initialization of the model
        # causing high uncertainty
        need_veto, reasoning, risk_reason = self.veto.assess_action(state, action)
        
        # We can't assert the exact outcome since it depends on the random model initialization,
        # but we can check that the reasoning contains "uncertainty" if veto is needed
        if need_veto:
            self.assertIn("uncertain", reasoning.lower())

if __name__ == '__main__':
    unittest.main()