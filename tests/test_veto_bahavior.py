import unittest
import numpy as np
import torch
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ai.models import DuelingDQN
from src.ai.uncertainty import UncertaintyEstimator
from src.veto.veto_mechanism import ThresholdVetoMechanism, UncertaintyVetoMechanism
from src.veto.learning_veto import LearningVetoMechanism
from src.veto.learning_risk_assessor import LearningRiskAssessor

class VetoBehaviorTest(unittest.TestCase):
    """Test the behavior of veto mechanisms in various scenarios."""
    
    def setUp(self):
        """Set up test environment with models and states."""
        # Create small models for testing
        self.state_size = 10  # Small for testing
        self.action_size = 4  # Up, Down, Left, Right for simplicity
        
        # Create model and uncertainty estimator
        self.model = DuelingDQN(self.state_size, self.action_size)
        self.uncertainty_estimator = UncertaintyEstimator(self.model)
        
        # Create veto mechanisms
        self.threshold_veto = ThresholdVetoMechanism(threshold=0.5)
        self.uncertainty_veto = UncertaintyVetoMechanism(
            self.uncertainty_estimator, uncertainty_threshold=0.5
        )
        self.learning_veto = LearningVetoMechanism(
            self.uncertainty_estimator, enable_learning=True
        )
        
        # Create test states
        self.normal_state = np.zeros(self.state_size)
        self.normal_state[-3:] = [0.8, 0.8, 0.8]  # Good health, ammo, shields
        
        self.risky_state = np.zeros(self.state_size)
        self.risky_state[-3:] = [0.1, 0.1, 0.0]  # Low health, ammo, no shields
        
        # Create predictable Q-values
        self.balanced_q = np.array([0.5, 0.5, 0.5, 0.5])
        self.strong_preference_q = np.array([0.9, 0.1, 0.1, 0.1])
        self.all_bad_q = np.array([-0.1, -0.2, -0.3, -0.4])

    def test_threshold_veto_consistency(self):
        """Test that threshold veto gives consistent results for the same input."""
        # Same state, action, and Q-values should give same result
        for _ in range(5):  # Test multiple times for consistency
            decision1 = self.threshold_veto.assess_action(self.normal_state, 0, self.balanced_q)
            decision2 = self.threshold_veto.assess_action(self.normal_state, 0, self.balanced_q)
            self.assertEqual(decision1.vetoed, decision2.vetoed)
    
    def test_threshold_veto_low_resources(self):
        """Test threshold veto correctly identifies risky actions with low resources."""
        # Combat action with low ammo should be vetoed
        combat_action = 2  # Arbitrary combat action
        decision = self.threshold_veto.assess_action(self.risky_state, combat_action, self.balanced_q)
        
        # Should veto due to low resources
        self.assertTrue(decision.vetoed)
        self.assertIn("ammo", decision.reason.lower() or "")
        
    def test_uncertainty_veto_high_uncertainty(self):
        """Test uncertainty veto triggers on high uncertainty."""
        # Create a state that should have high uncertainty
        uncertain_state = self.normal_state.copy()
        
        # Mock the uncertainty estimator to return high uncertainty
        original_get_action_uncertainty = self.uncertainty_estimator.get_action_uncertainty
        
        try:
            # Override with mock that returns high uncertainty
            self.uncertainty_estimator.get_action_uncertainty = lambda s, a: 0.8
            
            # Check if veto is triggered
            decision = self.uncertainty_veto.assess_action(uncertain_state, 0, self.balanced_q)
            self.assertTrue(decision.vetoed)
            self.assertIn("uncertain", decision.reason.lower() or "")
        finally:
            # Restore original method
            self.uncertainty_estimator.get_action_uncertainty = original_get_action_uncertainty
            
    def test_uncertainty_veto_low_uncertainty(self):
        """Test uncertainty veto doesn't trigger on low uncertainty."""
        # Mock the uncertainty estimator to return low uncertainty
        original_get_action_uncertainty = self.uncertainty_estimator.get_action_uncertainty
        
        try:
            # Override with mock that returns low uncertainty
            self.uncertainty_estimator.get_action_uncertainty = lambda s, a: 0.2
            
            # Check if veto is not triggered (for normal state and decent Q-values)
            decision = self.uncertainty_veto.assess_action(self.normal_state, 0, self.balanced_q)
            self.assertFalse(decision.vetoed)
        finally:
            # Restore original method
            self.uncertainty_estimator.get_action_uncertainty = original_get_action_uncertainty
            
    def test_learning_veto_basic(self):
        """Test basic functionality of learning veto mechanism."""
        # Should start with default behavior similar to threshold veto
        decision = self.learning_veto.assess_action(self.risky_state, 2, self.balanced_q)
        self.assertTrue(decision.vetoed)  # Should veto risky action
        
        # Record some outcomes to learn from
        self.learning_veto.record_veto_decision(
            self.risky_state, 2, True, 1, 0.5  # Vetoed and got positive reward
        )
        
        # Learning rate is too slow to see immediate changes in behavior,
        # but we can verify the learning data is stored
        self.assertGreater(len(self.learning_veto.recent_decisions), 0)
        
    def test_learning_veto_improves_with_experience(self):
        """Test that learning veto improves with experience."""
        # Create experience for risk assessor
        risk_assessor = self.learning_veto.risk_assessor
        
        # Feed some experiences
        # Format: (features, is_risky)
        for _ in range(50):
            # Good outcome for action 0 in normal state
            features = risk_assessor.extract_features(self.normal_state, 0, self.balanced_q)
            risk_assessor.experience_buffer.append((features, False))  # Not risky
            
            # Bad outcome for action 1 in risky state
            features = risk_assessor.extract_features(self.risky_state, 1, self.balanced_q)
            risk_assessor.experience_buffer.append((features, True))  # Risky
        
        # Train the model
        risk_assessor.train_model()
        
        # Now test if it learned
        decision_normal = self.learning_veto.assess_action(self.normal_state, 0, self.balanced_q)
        decision_risky = self.learning_veto.assess_action(self.risky_state, 1, self.balanced_q)
        
        # Should have learned that action 0 in normal state is safe
        self.assertFalse(decision_normal.vetoed)
        
        # Should have learned that action 1 in risky state is risky
        self.assertTrue(decision_risky.vetoed)
    
    def test_consecutive_veto_prevention(self):
        """Test that veto mechanisms prevent excessive consecutive vetoes."""
        # Configure for testing
        self.uncertainty_veto.max_consecutive_vetoes = 3
        
        # Mock to always return high uncertainty
        original_get_action_uncertainty = self.uncertainty_estimator.get_action_uncertainty
        self.uncertainty_estimator.get_action_uncertainty = lambda s, a: 0.9
        
        try:
            # First 3 should trigger veto
            for _ in range(3):
                decision = self.uncertainty_veto.assess_action(self.normal_state, 0, self.balanced_q)
                self.assertTrue(decision.vetoed)
            
            # 4th should not veto to prevent excessive consecutive vetoes
            decision = self.uncertainty_veto.assess_action(self.normal_state, 0, self.balanced_q)
            self.assertFalse(decision.vetoed)
        finally:
            # Restore original method
            self.uncertainty_estimator.get_action_uncertainty = original_get_action_uncertainty
            
    def test_veto_risk_explanation(self):
        """Test that veto decisions provide meaningful explanations."""
        # Check that threshold veto gives explanations
        decision = self.threshold_veto.assess_action(self.risky_state, 2, self.all_bad_q)
        self.assertTrue(decision.vetoed)
        self.assertNotEqual(decision.reason, "")
        
        # Check that learning veto gives explanations
        decision = self.learning_veto.assess_action(self.risky_state, 2, self.all_bad_q)
        self.assertTrue(decision.vetoed)
        self.assertNotEqual(decision.reason, "")
        
    def test_q_value_based_veto(self):
        """Test that vetos consider Q-values appropriately."""
        # Threshold veto should consider very negative Q-values as risky
        decision = self.threshold_veto.assess_action(self.normal_state, 3, self.all_bad_q)
        self.assertTrue(decision.vetoed)
        
        # Strong preference for action 0 should not trigger veto for action 0
        decision = self.threshold_veto.assess_action(self.normal_state, 0, self.strong_preference_q)
        self.assertFalse(decision.vetoed)
        
        # But it might trigger veto for other actions
        decision = self.threshold_veto.assess_action(self.normal_state, 1, self.strong_preference_q)
        # Note: We don't assert specific behavior here as implementation may vary
        
    def test_veto_decision_properties(self):
        """Test that veto decisions contain necessary properties."""
        decision = self.uncertainty_veto.assess_action(self.normal_state, 0, self.balanced_q)
        
        # Basic properties all veto decisions should have
        self.assertIsNotNone(decision.original_action)
        self.assertIsNotNone(decision.vetoed)
        self.assertIsNotNone(getattr(decision, 'reason', None))
        
        # Uncertainty veto specific properties
        self.assertIsNotNone(getattr(decision, 'uncertainty', None))
        
    def test_veto_performance_metrics(self):
        """Test that learning veto maintains performance metrics."""
        # Record some veto decisions
        self.learning_veto.record_veto_decision(
            self.risky_state, 1, True, 2, 1.0  # Vetoed and good outcome
        )
        self.learning_veto.record_veto_decision(
            self.risky_state, 2, True, 3, -0.5  # Vetoed but bad outcome
        )
        
        # Get metrics
        metrics = self.learning_veto.get_metrics()
        
        # Should track count
        self.assertEqual(metrics["veto_count"], 2)
        
        # Should track successes
        self.assertGreaterEqual(metrics["successful_veto_count"], 0)
        
        # Should have success rate
        self.assertGreaterEqual(metrics["successful_veto_rate"], 0.0)
        self.assertLessEqual(metrics["successful_veto_rate"], 1.0)
        
    def test_veto_with_actual_model_output(self):
        """Test veto mechanisms with actual model output, not mocked values."""
        # Get actual Q-values from model
        state_tensor = torch.FloatTensor(self.normal_state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).numpy()[0]
        
        # Get actual uncertainty
        uncertainty = self.uncertainty_estimator.estimate_uncertainty(self.normal_state)
        
        # Test threshold veto
        decision = self.threshold_veto.assess_action(self.normal_state, 0, q_values)
        # We don't assert specific behavior, just that it runs without error
        
        # Test uncertainty veto with real uncertainty
        decision = self.uncertainty_veto.assess_action(self.normal_state, 0, q_values)
        # Again, just test it runs without error

if __name__ == '__main__':
    unittest.main()