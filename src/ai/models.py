import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DuelingDQN(nn.Module):
    """
    Simplified Dueling DQN architecture that separates state value and action advantages.
    This helps the model learn which states are valuable without having to learn
    the effect of each action for each state.
    """
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        
        # Store parameters
        self.state_size = state_size
        self.action_size = action_size
        
        # Feature extraction layers with dropout for uncertainty estimation
        self.features = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Add dropout for MC dropout uncertainty estimation
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2)   # Add dropout for MC dropout uncertainty estimation
        )
        
        # Value stream - estimates state value
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream - estimates advantage of each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
            
        # Add batch dimension if necessary
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        features = self.features(x)
        
        # Value and advantage streams
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values

class EnsembleDQN:
    """
    Ensemble of DQN models for uncertainty estimation.
    Uses disagreement between models as a measure of epistemic uncertainty.
    """
    def __init__(self, state_size, action_size, ensemble_size=3):
        self.ensemble_size = ensemble_size
        self.state_size = state_size
        self.action_size = action_size
        self.models = [DuelingDQN(state_size, action_size) for _ in range(ensemble_size)]
        
    def forward(self, x):
        """Get predictions from all models in the ensemble"""
        predictions = []
        for model in self.models:
            q_values = model(x)
            predictions.append(q_values)
        return torch.stack(predictions)
        
    def predict(self, x):
        """Get mean prediction across the ensemble"""
        with torch.no_grad():
            predictions = self.forward(x)
            return predictions.mean(dim=0)
            
    def uncertainty(self, x):
        """Calculate uncertainty as disagreement between models"""
        with torch.no_grad():
            predictions = self.forward(x)
            # Standard deviation across models
            return predictions.std(dim=0)