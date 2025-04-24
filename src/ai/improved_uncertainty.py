"""
Improved uncertainty estimation module.
Provides multiple methods for estimating uncertainty in model predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from collections import defaultdict
from src.game.state import GameState
from src.game.state_adapter import StateAdapter

class ImprovedUncertaintyEstimator:
    """
    Enhanced uncertainty estimator with multiple estimation methods
    and better calibration.
    """
    def __init__(self, model, num_models=5, mc_dropout_samples=20):
        """
        Initialize uncertainty estimator
        
        Args:
            model: Base model to use for uncertainty estimation
            num_models: Number of models for ensemble method
            mc_dropout_samples: Number of samples for MC dropout
        """
        self.base_model = model
        self.mc_dropout_samples = mc_dropout_samples
        
        # Create ensemble models if using ensemble method
        self.ensemble_models = []
        self.use_ensemble = num_models > 1
        if self.use_ensemble:
            # Create ensemble of models with same architecture but different initializations
            for i in range(num_models):
                ensemble_model = self._clone_model_architecture(model)
                # Apply different initialization to ensure diversity
                self._init_model_weights(ensemble_model, i)
                self.ensemble_models.append(ensemble_model)
        
        # Cache for uncertainty estimates
        self.estimate_cache = {}
        self.cache_timeout = 5.0  # seconds
        
        # Calibration parameters
        self.calibration = {
            'mc_dropout': {
                'scale': 1.0,
                'offset': 0.0
            },
            'ensemble': {
                'scale': 1.0,
                'offset': 0.0
            },
            'combined': {
                'mc_weight': 0.5,
                'ensemble_weight': 0.5
            }
        }
        
        # Usage statistics
        self.stats = defaultdict(int)
    
    def _clone_model_architecture(self, model):
        """Clone model architecture without copying weights"""
        if hasattr(model, 'state_size') and hasattr(model, 'action_size'):
            # If model has these attributes, assume it's a DuelingDQN
            return type(model)(model.state_size, model.action_size)
        else:
            # Generic approach - might not work for all models
            return type(model)(*getattr(model, '__init__args__', []), **getattr(model, '__init__kwargs__', {}))
    
    def _init_model_weights(self, model, seed=None):
        """Initialize model weights with different random seed"""
        if seed is not None:
            torch.manual_seed(seed)
            
        # Initialize each layer with a different distribution
        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'conv' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'batch' in name:
                    nn.init.constant_(param, 1)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def estimate_uncertainty(self, state, method='combined'):
        """
        Estimate uncertainty for all actions
        
        Args:
            state: State to estimate uncertainty for
            method: Uncertainty estimation method ('mc_dropout', 'ensemble', 'combined')
            
        Returns:
            Uncertainty values for each action
        """
        # Convert to GameState if needed
        if not isinstance(state, GameState):
            try:
                state = GameState(raw_state=state)
            except:
                # If conversion fails, proceed with raw state
                pass
            
        # Check cache
        cache_key = self._create_cache_key(state, method)
        current_time = time.time()
        
        if cache_key in self.estimate_cache:
            entry_time, uncertainty = self.estimate_cache[cache_key]
            if current_time - entry_time < self.cache_timeout:
                self.stats['cache_hits'] += 1
                return uncertainty
                
        self.stats['cache_misses'] += 1
        
        # Estimate uncertainty using the specified method
        if method == 'mc_dropout':
            uncertainty = self._mc_dropout_uncertainty(state)
        elif method == 'ensemble':
            uncertainty = self._ensemble_uncertainty(state)
        elif method == 'combined':
            uncertainty = self._combined_uncertainty(state)
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
            
        # Apply calibration
        uncertainty = self._calibrate_uncertainty(uncertainty, method)
        
        # Update cache
        self.estimate_cache[cache_key] = (current_time, uncertainty)
        
        return uncertainty
    
    def get_action_uncertainty(self, state, action, method='combined'):
        """
        Get uncertainty for a specific action
        
        Args:
            state: State to estimate uncertainty for
            action: Action to get uncertainty for
            method: Uncertainty estimation method
            
        Returns:
            Uncertainty value for the action
        """
        uncertainty_values = self.estimate_uncertainty(state, method)
        
        # Handle case where action is out of bounds
        if action >= len(uncertainty_values):
            # Return max uncertainty as fallback
            return np.max(uncertainty_values)
            
        return uncertainty_values[action]
    
    def _mc_dropout_uncertainty(self, state):
        """
        Estimate uncertainty using Monte Carlo dropout
        
        Args:
            state: State to estimate uncertainty for
            
        Returns:
            Uncertainty values for each action
        """
        # Convert state for model input
        if isinstance(state, GameState):
            state_tensor = state.to_tensor().unsqueeze(0)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
        # Set model to training mode to enable dropout
        self.base_model.train()
        
        # Perform multiple forward passes
        samples = []
        with torch.no_grad():
            for _ in range(self.mc_dropout_samples):
                output = self.base_model(state_tensor)
                samples.append(output)
                
        # Set model back to evaluation mode
        self.base_model.eval()
        
        # Calculate uncertainty (standard deviation across samples)
        samples = torch.cat(samples, dim=0)
        uncertainty = samples.std(dim=0).squeeze().numpy()
        
        # Apply calibration parameters
        params = self.calibration['mc_dropout']
        uncertainty = uncertainty * params['scale'] + params['offset']
        
        # Normalize to [0,1] for easier interpretation
        if np.max(uncertainty) > 0:
            uncertainty = uncertainty / np.max(uncertainty)
            
        return uncertainty
    
    def _ensemble_uncertainty(self, state):
        """
        Estimate uncertainty using ensemble disagreement
        
        Args:
            state: State to estimate uncertainty for
            
        Returns:
            Uncertainty values for each action
        """
        if not self.use_ensemble:
            # Fall back to MC dropout if ensemble not available
            return self._mc_dropout_uncertainty(state)
            
        # Convert state for model input
        if isinstance(state, GameState):
            state_tensor = state.to_tensor().unsqueeze(0)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
        # Get predictions from all ensemble models
        predictions = []
        with torch.no_grad():
            for model in self.ensemble_models:
                output = model(state_tensor)
                predictions.append(output)
                
        # Calculate uncertainty (standard deviation across models)
        predictions = torch.cat(predictions, dim=0)
        uncertainty = predictions.std(dim=0).squeeze().numpy()
        
        # Apply calibration parameters
        params = self.calibration['ensemble']
        uncertainty = uncertainty * params['scale'] + params['offset']
        
        # Normalize to [0,1] for easier interpretation
        if np.max(uncertainty) > 0:
            uncertainty = uncertainty / np.max(uncertainty)
            
        return uncertainty
    
    def _combined_uncertainty(self, state):
        """
        Combine multiple uncertainty estimation methods
        
        Args:
            state: State to estimate uncertainty for
            
        Returns:
            Combined uncertainty values for each action
        """
        # Get uncertainty estimates from different methods
        mc_uncertainty = self._mc_dropout_uncertainty(state)
        
        if self.use_ensemble:
            ensemble_uncertainty = self._ensemble_uncertainty(state)
            
            # Combine using weighted average
            weights = self.calibration['combined']
            uncertainty = (
                mc_uncertainty * weights['mc_weight'] +
                ensemble_uncertainty * weights['ensemble_weight']
            )
        else:
            # Fall back to MC dropout if ensemble not available
            uncertainty = mc_uncertainty
            
        return uncertainty
    
    def _calibrate_uncertainty(self, uncertainty, method):
        """
        Apply calibration to uncertainty estimates
        
        Args:
            uncertainty: Raw uncertainty values
            method: Uncertainty method used
            
        Returns:
            Calibrated uncertainty values
        """
        # Apply simple scaling and offset calibration
        if method in self.calibration:
            params = self.calibration[method]
            if isinstance(params, dict) and 'scale' in params:
                uncertainty = uncertainty * params['scale'] + params.get('offset', 0.0)
                
        # Ensure values are in [0,1] range
        uncertainty = np.clip(uncertainty, 0.0, 1.0)
        
        return uncertainty
    
    def update_calibration(self, true_uncertainty, estimated_uncertainty, method):
        """
        Update calibration parameters based on observed data
        
        Args:
            true_uncertainty: True uncertainty values (from validation)
            estimated_uncertainty: Estimated uncertainty values
            method: Uncertainty method to calibrate
            
        Returns:
            Updated calibration parameters
        """
        if method not in self.calibration:
            return
            
        # Fit simple linear calibration parameters
        # uncertainty_true = scale * uncertainty_estimated + offset
        if len(true_uncertainty) != len(estimated_uncertainty):
            raise ValueError("True and estimated uncertainty must have same length")
            
        # Use least squares to find optimal parameters
        X = np.column_stack([estimated_uncertainty, np.ones_like(estimated_uncertainty)])
        y = true_uncertainty
        
        try:
            # Solve for parameters
            scale, offset = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Update calibration
            if method in ['mc_dropout', 'ensemble']:
                self.calibration[method]['scale'] = float(scale)
                self.calibration[method]['offset'] = float(offset)
            
            return {'scale': float(scale), 'offset': float(offset)}
        except:
            # If fitting fails, leave calibration unchanged
            return self.calibration[method]
    
    def update_weights(self, method_scores):
        """
        Update method weights based on their performance
        
        Args:
            method_scores: Dictionary mapping methods to their scores
            
        Returns:
            Updated weights
        """
        if 'combined' not in self.calibration:
            return
            
        total_score = sum(method_scores.values())
        if total_score <= 0:
            return
            
        # Normalize scores to sum to 1
        normalized_scores = {k: v / total_score for k, v in method_scores.items()}
        
        # Update weights
        if 'mc_dropout' in normalized_scores:
            self.calibration['combined']['mc_weight'] = normalized_scores['mc_dropout']
            
        if 'ensemble' in normalized_scores:
            self.calibration['combined']['ensemble_weight'] = normalized_scores['ensemble']
            
        # Ensure weights sum to 1
        total_weight = (
            self.calibration['combined']['mc_weight'] +
            self.calibration['combined']['ensemble_weight']
        )
        
        if total_weight > 0:
            self.calibration['combined']['mc_weight'] /= total_weight
            self.calibration['combined']['ensemble_weight'] /= total_weight
            
        return self.calibration['combined']
    
    def _create_cache_key(self, state, method):
        """Create cache key for uncertainty estimates"""
        # Use state values and method as key
        if isinstance(state, GameState):
            state_array = state.raw
        else:
            state_array = state
            
        # Use first few and last few state values for cache key
        # This is a compromise between precision and cache efficiency
        key_indices = list(range(min(10, len(state_array)))) + list(range(max(0, len(state_array) - 3), len(state_array)))
        key_values = [state_array[i] for i in key_indices]
        
        # Round values to reduce cache size
        key_values = [float(np.round(v, 3)) for v in key_values]
        
        # Create key from state values and method
        return (method, tuple(key_values))
    
    def reset_cache(self):
        """Clear the uncertainty cache"""
        self.estimate_cache = {}
        
    def get_stats(self):
        """Get usage statistics"""
        return dict(self.stats)
    
    def validate_calibration(self, validation_data):
        """
        Validate uncertainty calibration against ground truth data
        
        Args:
            validation_data: List of (state, action, ground_truth_uncertainty) tuples
            
        Returns:
            Validation metrics
        """
        if not validation_data:
            return {}
            
        # Extract data
        states, actions, true_uncertainties = zip(*validation_data)
        
        # Get estimated uncertainties for each method
        method_results = {}
        
        for method in ['mc_dropout', 'ensemble', 'combined']:
            if method == 'ensemble' and not self.use_ensemble:
                continue
                
            # Get estimated uncertainties
            estimated = []
            for state, action in zip(states, actions):
                estimated.append(self.get_action_uncertainty(state, action, method))
                
            # Calculate metrics
            mse = np.mean((np.array(true_uncertainties) - np.array(estimated))**2)
            correlation = np.corrcoef(true_uncertainties, estimated)[0, 1]
            
            method_results[method] = {
                'mse': mse,
                'correlation': correlation,
                'estimated': estimated
            }
            
        # Update calibration if validation data is sufficient
        if len(validation_data) >= 30:
            for method in method_results:
                self.update_calibration(
                    true_uncertainties,
                    method_results[method]['estimated'],
                    method
                )
                
            # Update method weights
            method_scores = {m: 1.0 / (r['mse'] + 1e-6) for m, r in method_results.items()}
            self.update_weights(method_scores)
            
        return {
            'method_results': method_results,
            'calibration': self.calibration
        }
    
    def uncertainty_to_risk(self, uncertainty):
        """
        Convert uncertainty estimate to risk score
        
        Args:
            uncertainty: Uncertainty value in [0,1]
            
        Returns:
            Risk score in [0,1]
        """
        # Simple sigmoidal mapping from uncertainty to risk
        # This can be replaced with a more sophisticated mapping based on validation
        if uncertainty < 0.2:
            # Low uncertainty -> very low risk
            return uncertainty * 0.25  # Max 0.05
        elif uncertainty < 0.5:
            # Medium uncertainty -> moderate risk
            return 0.05 + (uncertainty - 0.2) * 0.5  # 0.05 to 0.2
        else:
            # High uncertainty -> high risk, progressively increasing
            return 0.2 + (uncertainty - 0.5) * 1.6  # 0.2 to 1.0
    
    def expected_regret(self, state, action, q_values=None):
        """
        Estimate expected regret for an action based on uncertainty
        
        Args:
            state: Current state
            action: Action to evaluate
            q_values: Optional Q-values for state
            
        Returns:
            Expected regret estimate
        """
        # Get uncertainty for the action
        uncertainty = self.get_action_uncertainty(state, action)
        
        # If Q-values are available, use them to improve regret estimation
        if q_values is not None:
            # Get Q-value for this action and best Q-value
            action_q = q_values[action]
            best_q = np.max(q_values)
            
            # Calculate Q-value gap (how far from optimal)
            q_gap = max(0, best_q - action_q)
            
            # Combine uncertainty and Q-gap for regret estimate
            # High uncertainty + large Q-gap = high regret
            regret = (uncertainty * 0.5) + (q_gap * 0.5)
            
            # Normalize to [0,1]
            regret = min(1.0, regret)
        else:
            # Without Q-values, use uncertainty as proxy for regret
            regret = self.uncertainty_to_risk(uncertainty)
            
        return regret