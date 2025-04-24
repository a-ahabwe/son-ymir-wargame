import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.ai.models import EnsembleDQN

class UncertaintyEstimator:
    """
    Class for estimating uncertainty in model predictions.
    
    Enhanced with:
    1. Better distinction between aleatoric and epistemic uncertainty
    2. Improved uncertainty propagation methods
    3. Bayesian uncertainty quantification
    4. Combined ensemble and MC dropout approaches
    """
    def __init__(self, model, action_size=11, ensemble_size=3, mc_dropout_samples=10, distributional=False):
        self.model = model
        self.action_size = action_size
        self.ensemble = None
        self.mc_dropout_samples = mc_dropout_samples
        self.distributional = distributional
        self.uncertainty_history = []  # Store recent uncertainty estimates
        self.max_history = 100
        self.ensemble_size = ensemble_size
        
        # Create mini-ensemble by using MC dropout multiple times
        # This combines the benefits of both approaches
        self.use_hybrid_uncertainty = True
        
        # For adaptive thresholding
        self.uncertainty_threshold_history = []
        self.adaptive_threshold_window = 50
        
        # Normalize uncertainty to more usable range
        self.normalize_uncertainty = True
        
        # Use all available information for uncertainty
        if distributional and hasattr(model, 'get_q_dist'):
            self.use_distributional_uncertainty = True
        else:
            self.use_distributional_uncertainty = False
        
        # Try to get it from model attributes, or from model architecture
        if hasattr(model, 'action_size'):
            self.action_size = model.action_size
        elif hasattr(model, 'advantage_stream') and isinstance(model.advantage_stream, torch.nn.Sequential):
            # Try to get it from the last layer of the advantage stream (for DuelingDQN)
            last_layer = model.advantage_stream[-1]
            if hasattr(last_layer, 'out_features'):
                self.action_size = last_layer.out_features
                if distributional and hasattr(model, 'num_atoms'):
                    self.action_size = self.action_size // model.num_atoms
            else:
                # Default fallback
                self.action_size = 11  # Based on your environment's action space
        else:
            # Default fallback
            self.action_size = 11  # Based on your environment's action space
        
        # For Monte Carlo dropout method, we use the provided model
        # For ensemble method, we create a separate ensemble
        if ensemble_size > 1:
            # Clone model architecture to create ensemble
            if hasattr(model, 'state_size') and hasattr(model, 'action_size'):
                self.ensemble = EnsembleDQN(model.state_size, model.action_size, ensemble_size, distributional)
                
    def epistemic_uncertainty(self, state):
        """
        Estimate epistemic uncertainty (model uncertainty)
        Returns uncertainty score for each action
        
        Epistemic uncertainty represents uncertainty due to limited knowledge/data
        """
        if self.use_hybrid_uncertainty:
            # Combine multiple approaches for more accurate estimation
            return self._hybrid_uncertainty(state)
        elif self.ensemble is not None:
            # Ensemble disagreement method
            return self._ensemble_uncertainty(state)
        else:
            # Monte Carlo dropout method
            return self._dropout_uncertainty(state)
            
    def _hybrid_uncertainty(self, state):
        """
        Combined uncertainty estimation using both MC dropout and ensemble-like approaches
        This creates a more robust uncertainty estimate
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        # Enable dropout layers for inference (MC dropout)
        self.model.train()
        
        # Perform multiple forward passes, treating them as ensemble members
        samples = []
        for _ in range(self.mc_dropout_samples):
            with torch.no_grad():
                if self.distributional and hasattr(self.model, 'get_q_values'):
                    output = self.model.get_q_values(state)
                else:
                    output = self.model(state)
                samples.append(output)
                
        # Disable dropout (back to evaluation mode)
        self.model.eval()
        
        # Calculate variance of predictions (epistemic uncertainty)
        samples = torch.stack(samples, dim=0)
        uncertainty = samples.std(dim=0).numpy()
        
        # Context-aware uncertainty scaling based on action types
        # Get action categories - state format is assumed to be grid + agent stats
        action_uncertainty_weights = self._get_context_aware_uncertainty_weights(state)
        
        # Apply context-aware scaling instead of uniform scaling
        uncertainty = uncertainty * action_uncertainty_weights
        
        # Normalize uncertainty to [0,1] range for easier thresholding
        if self.normalize_uncertainty and np.max(uncertainty) > 0:
            uncertainty = uncertainty / np.max(uncertainty)
            # Apply additional sigmoid-like squashing to further normalize
            uncertainty = 1.0 / (1.0 + np.exp(-5 * (uncertainty - 0.5)))
            
        return uncertainty
        
    def _get_context_aware_uncertainty_weights(self, state):
        """
        Generate context-aware weights for different action types based on the state
        
        This replaces hard-coded scaling factors with a more adaptive approach
        """
        # Extract agent stats from state (last 3 elements - health, ammo, shields)
        if isinstance(state, torch.Tensor):
            state_np = state.cpu().numpy()
            if len(state_np.shape) > 1 and state_np.shape[0] > 1:
                state_np = state_np[0]  # Take first batch element if batched
        else:
            state_np = state
            
        # Get agent stats - assuming last 3 elements are health, ammo, shields
        agent_health = state_np[-3]
        agent_ammo = state_np[-2]
        agent_shields = state_np[-1]
        
        # Increase base weights to generate higher uncertainty values
        movement_weight = 0.9  # Increased from 0.6 - higher uncertainty for movement
        combat_weight = 1.2    # Increased from 1.0 - higher uncertainty for combat
        special_weight = 1.0   # Increased from 0.8 - higher uncertainty for special actions
        
        # Adjust weights based on agent context 
        # When health is low, movement becomes more critical (reduce uncertainty threshold)
        if agent_health < 0.3:
            movement_weight *= 0.9  # Less reduction than before (was 0.8)
            combat_weight *= 1.4    # Higher increase (was 1.2)
        
        # When ammo is low, combat actions should have higher uncertainty
        if agent_ammo < 0.2:
            combat_weight *= 1.5  # Increased from 1.3
            
        # Create weight array for all actions
        # Assuming actions 0-3 are movement, 4-7 are combat, 8-10 are special
        weights = np.ones(self.action_size)
        
        # Apply weights by action category
        weights[0:4] = movement_weight  # Movement actions
        if len(weights) > 7:  # Ensure array is large enough
            weights[4:8] = combat_weight   # Combat actions
        if len(weights) > 10:  # Ensure array is large enough
            weights[8:11] = special_weight  # Special actions
            
        return weights
        
    def _ensemble_uncertainty(self, state):
        """Estimate uncertainty using ensemble disagreement"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        # Get uncertainty from ensemble
        return self.ensemble.uncertainty(state).numpy()
        
    def _dropout_uncertainty(self, state):
        """
        Estimate uncertainty using Monte Carlo dropout
        This is an approximation of Bayesian inference with NNs
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        # Enable dropout layers for inference
        self.model.train()
        
        # Perform multiple forward passes
        samples = []
        for _ in range(self.mc_dropout_samples):  # Use more samples for better estimation
            with torch.no_grad():
                if self.distributional and hasattr(self.model, 'get_q_values'):
                    output = self.model.get_q_values(state)
                else:
                    output = self.model(state)
                samples.append(output)
                
        # Disable dropout (back to evaluation mode)
        self.model.eval()
        
        # Calculate standard deviation across samples
        samples = torch.stack(samples, dim=0)
        return samples.std(dim=0).numpy()
        
    def aleatoric_uncertainty(self, state):
        """
        Estimate aleatoric uncertainty (data uncertainty)
        
        Aleatoric uncertainty represents inherent noise/randomness in the environment
        """
        if self.distributional and hasattr(self.model, 'get_q_dist'):
            # For distributional models, use the variance of the value distribution
            try:
                with torch.no_grad():
                    if not isinstance(state, torch.Tensor):
                        state = torch.FloatTensor(state)
                    
                    # Get the full distribution
                    q_dist = self.model.get_q_dist(state)
                    
                    # Calculate variance of the distribution for each action
                    # Shape: (batch_size, action_size)
                    means = torch.sum(q_dist * self.model.support.expand_as(q_dist), dim=2)
                    
                    # Calculate second moment E[X²]
                    sq_means = torch.sum(q_dist * (self.model.support.expand_as(q_dist)**2), dim=2)
                    
                    # Variance = E[X²] - E[X]²
                    variances = sq_means - means**2
                    
                    return variances.numpy() * 0.7  # Scale down
            except (AttributeError, Exception) as e:
                # Fallback if something goes wrong
                return np.ones(self.action_size) * 0.05
        else:
            # For non-distributional models, use a lower default uncertainty
            return np.ones(self.action_size) * 0.05
    
    def bayesian_uncertainty(self, state):
        """
        Comprehensive Bayesian uncertainty estimation combining both
        aleatoric and epistemic components
        """
        epistemic = self.epistemic_uncertainty(state)
        aleatoric = self.aleatoric_uncertainty(state)
        
        # Total uncertainty is sum of both components with weightings
        # Reduce the weight of aleatoric uncertainty to lower total uncertainty
        return epistemic * 0.8 + aleatoric * 0.2
        
    def adaptive_uncertainty_threshold(self):
        """
        Dynamically adapt uncertainty threshold based on recent observations
        This helps the system adjust to the current uncertainty distribution
        """
        if len(self.uncertainty_threshold_history) < self.adaptive_threshold_window:
            # Not enough data yet, use default
            return 0.4  # Increased from 0.3
            
        # Calculate threshold based on recent uncertainty values
        # Use percentile approach - threshold at 80th percentile (increased from 70th)
        uncertainties = np.array(self.uncertainty_threshold_history)
        return np.percentile(uncertainties, 80)
        
    def decision_uncertainty(self, state, action):
        """Calculate overall uncertainty for a specific action"""
        # TESTING: Use fixed values for uncertainty to test veto mechanism
        fixed_uncertainty = 0.05  # Base uncertainty level
        
        if action in [2, 5, 8]:  # Left, Shoot Down, Reload - high uncertainty
            fixed_uncertainty = 0.80  # Significantly above typical threshold
        elif action in [4, 7]:  # Shoot Up, Shoot Right - medium uncertainty
            fixed_uncertainty = 0.40  # Above threshold
        else:
            fixed_uncertainty = 0.08  # Below threshold
            
        print(f"DEBUG: Using fixed uncertainty for action {action}: {fixed_uncertainty:.4f}")
        
        # Store in history for adaptive thresholding
        if len(self.uncertainty_threshold_history) >= self.adaptive_threshold_window:
            self.uncertainty_threshold_history.pop(0)
        self.uncertainty_threshold_history.append(fixed_uncertainty)
        
        return fixed_uncertainty
        
        # Original implementation (commented out for testing)
        """
        # Get Bayesian uncertainty estimates using hybrid approach
        uncertainty_values = self._hybrid_uncertainty(state)
        
        # Store in history for tracking and adaptation
        if len(self.uncertainty_history) >= self.max_history:
            self.uncertainty_history.pop(0)
        self.uncertainty_history.append(uncertainty_values)
        
        # Update threshold history
        if isinstance(uncertainty_values, np.ndarray):
            avg_uncertainty = np.mean(uncertainty_values)
            if len(self.uncertainty_threshold_history) >= self.adaptive_threshold_window:
                self.uncertainty_threshold_history.pop(0)
            self.uncertainty_threshold_history.append(avg_uncertainty)
        
        # Extract uncertainty for the specific action
        if isinstance(uncertainty_values, np.ndarray):
            # Handle different array shapes
            if len(uncertainty_values.shape) > 1:
                # If we have a 2D array (batch dimension), take the first row
                uncertainty_values = uncertainty_values[0]
                
            # If the array is smaller than our action space, extend it
            if len(uncertainty_values) <= action:
                # Create a new array of the right size
                extended_uncertainty = np.ones(self.action_size) * 0.3  # Reduced default
                # Copy existing values, handling potential shape mismatches
                extended_uncertainty[:min(len(extended_uncertainty), len(uncertainty_values))] = uncertainty_values[:min(len(extended_uncertainty), len(uncertainty_values))]
                uncertainty_values = extended_uncertainty
                
            # Return uncertainty value for requested action
            return uncertainty_values[action]
        else:
            # Return the scalar uncertainty if that's what we have
            return uncertainty_values
        """
        
    def get_uncertainty_map(self, state):
        """
        Generate an uncertainty map for all actions.
        This can be used for visualization or uncertainty-aware exploration.
        """
        uncertainty_map = self.bayesian_uncertainty(state)
        
        if isinstance(uncertainty_map, np.ndarray) and len(uncertainty_map.shape) > 1:
            uncertainty_map = uncertainty_map[0]
            
        # Reduce uncertainty reduction for movement - make movement more likely to trigger veto
        if isinstance(uncertainty_map, np.ndarray) and len(uncertainty_map) >= 4:
            uncertainty_map[0:4] = uncertainty_map[0:4] * 0.5  # Apply 50% reduction instead of 80%
            
        return uncertainty_map
        
    def adaptive_exploration(self, state):
        """
        Calculate exploration bonuses based on uncertainty.
        This can be used for uncertainty-driven exploration policies.
        """
        uncertainty_map = self.get_uncertainty_map(state)
        
        # Normalize to [0, 1] for bonus calculation
        if np.max(uncertainty_map) > 0:
            normalized_map = uncertainty_map / np.max(uncertainty_map)
        else:
            normalized_map = uncertainty_map
            
        return normalized_map