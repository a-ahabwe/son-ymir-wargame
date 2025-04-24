import torch
import torch.nn as nn
import numpy as np

class UncertaintyEstimator:
    """
    Simplified class for estimating uncertainty in model predictions.
    Uses Monte Carlo dropout as the primary uncertainty estimation method.
    """
    def __init__(self, model, mc_dropout_samples=10):
        self.model = model
        self.mc_dropout_samples = mc_dropout_samples
        
        # For adaptive thresholding
        self.uncertainty_history = []
        self.max_history = 100
        
    def estimate_uncertainty(self, state):
        """
        Estimate uncertainty using Monte Carlo dropout
        Returns uncertainty score for each action
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        # Add batch dimension if necessary
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        # Check if input size matches model's expected input size
        expected_size = self.model.state_size
        if state.shape[1] != expected_size:
            # Handle size mismatch by resizing the input
            print(f"Warning: Input size mismatch in uncertainty estimator. Got {state.shape[1]}, expected {expected_size}.")
            # Resize tensor by either padding with zeros or truncating
            if state.shape[1] < expected_size:
                # Pad with zeros
                padding = torch.zeros(state.shape[0], expected_size - state.shape[1])
                state = torch.cat([state, padding], dim=1)
            else:
                # Truncate
                state = state[:, :expected_size]
            
        # Enable dropout layers for inference (MC dropout)
        self.model.train()
        
        # Perform multiple forward passes
        samples = []
        for _ in range(self.mc_dropout_samples):
            with torch.no_grad():
                output = self.model(state)
                samples.append(output)
                
        # Disable dropout (back to evaluation mode)
        self.model.eval()
        
        # Calculate variance across samples (epistemic uncertainty)
        samples = torch.stack(samples, dim=0)
        uncertainty = samples.std(dim=0).numpy()
        
        # Normalize uncertainty to [0,1] range for easier thresholding
        if np.max(uncertainty) > 0:
            uncertainty = uncertainty / np.max(uncertainty)
            
        # Store in history for adaptive thresholding
        if len(self.uncertainty_history) >= self.max_history:
            self.uncertainty_history.pop(0)
        self.uncertainty_history.append(float(np.mean(uncertainty)))
        
        return uncertainty
        
    def get_action_uncertainty(self, state, action):
        """Calculate uncertainty for a specific action"""
        uncertainty_values = self.estimate_uncertainty(state)
        
        if isinstance(uncertainty_values, np.ndarray):
            # Handle different array shapes
            if len(uncertainty_values.shape) > 1:
                uncertainty_values = uncertainty_values[0]
                
            # Return uncertainty value for requested action
            if action < len(uncertainty_values):
                return uncertainty_values[action]
                
        # Default value if we couldn't get specific uncertainty
        return 0.2
        
    def get_adaptive_threshold(self):
        """
        Dynamically adapt uncertainty threshold based on recent observations
        """
        if len(self.uncertainty_history) < 10:
            # Not enough data yet, use default
            return 0.5
            
        # Calculate threshold based on recent uncertainty values
        # Use percentile approach - threshold at 75th percentile
        return np.percentile(self.uncertainty_history, 75)