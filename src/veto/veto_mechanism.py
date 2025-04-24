import time
import numpy as np
from src.veto.risk_assessment import RiskAssessor

class VetoDecision:
    """Simple container for veto decision information"""
    def __init__(self, original_action, vetoed, reason="", alternative_action=None, uncertainty=None):
        self.original_action = original_action
        self.vetoed = vetoed
        self.reason = reason
        self.alternative_action = alternative_action
        self.uncertainty = uncertainty
        self.timestamp = time.time()

class VetoMechanism:
    """Base class for veto mechanisms"""
    def __init__(self, timeout=10):
        self.timeout = timeout
        self.risk_assessor = RiskAssessor()
        self.veto_history = []
        
    def assess_action(self, state, action, q_values=None):
        """
        Determine if veto is needed (base implementation)
        Returns VetoDecision object
        """
        # Basic implementation just checks risk
        is_risky, risk_reason = self.risk_assessor.is_high_risk(state, action, q_values)
        
        return VetoDecision(
            original_action=action,
            vetoed=is_risky,
            reason=risk_reason if is_risky else ""
        )
            
    def record_veto_decision(self, state, action, vetoed, alternative=None, outcome=None):
        """Record veto decision and outcome for analysis"""
        self.veto_history.append({
            'state': state,
            'action': action,
            'vetoed': vetoed,
            'alternative': alternative,
            'outcome': outcome,
            'timestamp': time.time()
        })
        
class ThresholdVetoMechanism(VetoMechanism):
    """
    Threshold-based veto mechanism.
    Uses simple risk-based approach with a threshold.
    """
    def __init__(self, threshold=0.7, timeout=10):
        super().__init__(timeout)
        self.threshold = threshold
        
    def assess_action(self, state, action, q_values=None):
        """Determine if veto is needed using threshold on risk assessment"""
        is_risky, risk_reason = self.risk_assessor.is_high_risk(state, action, q_values)
        
        # Only veto if risk assessment exceeds threshold
        # The threshold is a simple parameter that controls veto frequency
        return VetoDecision(
            original_action=action,
            vetoed=is_risky and (q_values is None or q_values[action] < self.threshold),
            reason=risk_reason if is_risky else ""
        )
            
class UncertaintyVetoMechanism(VetoMechanism):
    """
    Uncertainty-based veto mechanism.
    Triggers veto when model uncertainty is high.
    """
    def __init__(self, uncertainty_estimator, uncertainty_threshold=0.5, timeout=10):
        super().__init__(timeout)
        self.uncertainty_estimator = uncertainty_estimator
        self.uncertainty_threshold = uncertainty_threshold
        self.consecutive_veto_count = 0
        self.max_consecutive_vetoes = 3
        
    def assess_action(self, state, action, q_values=None):
        """Determine if veto is needed based on uncertainty"""
        # First check basic risk
        is_risky, risk_reason = self.risk_assessor.is_high_risk(state, action, q_values)
        
        # Then get action uncertainty
        uncertainty = self.uncertainty_estimator.get_action_uncertainty(state, action)
        
        # Determine if we should veto based on uncertainty and risk
        should_veto = False
        veto_reason = ""
        
        if uncertainty > self.uncertainty_threshold:
            should_veto = True
            veto_reason = f"High uncertainty in action outcome ({uncertainty:.2f})"
        elif is_risky:
            should_veto = True
            veto_reason = risk_reason
            
        # Prevent excessive consecutive vetoes
        if should_veto:
            self.consecutive_veto_count += 1
            if self.consecutive_veto_count > self.max_consecutive_vetoes:
                # Allow the action through to prevent stalling
                should_veto = False
                veto_reason = "Veto suppressed to prevent excessive interventions"
        else:
            # Reset consecutive veto count
            self.consecutive_veto_count = 0
            
        # Return the decision
        return VetoDecision(
            original_action=action,
            vetoed=should_veto,
            reason=veto_reason,
            uncertainty=uncertainty
        )