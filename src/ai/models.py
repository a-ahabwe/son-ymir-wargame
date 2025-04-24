import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionModule(nn.Module):
    """
    Uncertainty-aware attention mechanism that focuses on important state features.
    This helps the model attend to regions with high decision influence.
    """
    def __init__(self, input_dim, uncertainty_guided=True):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Softmax(dim=1)
        )
        self.uncertainty_guided = uncertainty_guided
        
    def forward(self, x, uncertainty=None):
        # Calculate attention weights
        attention_weights = self.attention(x)
        
        # If uncertainty information is available and enabled, modulate attention
        if uncertainty is not None and self.uncertainty_guided:
            # Normalize uncertainty to [0, 1] range
            norm_uncertainty = uncertainty / (uncertainty.max() + 1e-8)
            # Increase attention to areas with high uncertainty
            attention_weights = attention_weights * (1 + norm_uncertainty)
            # Re-normalize
            attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True)
            
        # Apply attention to input
        attended_x = x * attention_weights
        return attended_x, attention_weights

class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture that separates state value and action advantages.
    This helps the model learn which states are valuable without having to learn
    the effect of each action for each state.
    
    Enhanced with:
    1. Attention mechanism to focus on important state features
    2. Distributional output option for C51 algorithm
    """
    def __init__(self, state_size, action_size, distributional=False, num_atoms=51, v_min=-10, v_max=10):
        super(DuelingDQN, self).__init__()
        
        # Store parameters
        self.state_size = state_size
        self.action_size = action_size
        self.distributional = distributional
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # For distributional RL
        if distributional:
            self.support = torch.linspace(v_min, v_max, num_atoms)
            self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Attention module
        self.attention = AttentionModule(state_size)
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Value stream - estimates state value
        if distributional:
            self.value_stream = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_atoms)  # Output for each atom
            )
        else:
            self.value_stream = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        
        # Advantage stream - estimates advantage of each action
        if distributional:
            self.advantage_stream = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_size * num_atoms)  # Output for each action-atom pair
            )
        else:
            self.advantage_stream = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_size)
            )
    
    def forward(self, x, uncertainty=None):
        """Forward pass through the network"""
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
            
        # Add batch dimension if necessary
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Apply attention mechanism
        x_attended, _ = self.attention(x, uncertainty)
            
        features = self.features(x_attended)
        
        if self.distributional:
            # Distributional DQN (C51)
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            
            # Reshape for action-distribution calculation
            advantages = advantages.view(-1, self.action_size, self.num_atoms)
            value = value.view(-1, 1, self.num_atoms)
            
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_dist = value + (advantages - advantages.mean(dim=1, keepdim=True))
            
            # Apply softmax to ensure valid probability distribution
            q_dist = F.softmax(q_dist, dim=2)
            
            # For compatibility with non-distributional code, also return mean values
            q_values = torch.sum(q_dist * self.support.expand_as(q_dist), dim=2)
            
            return q_values, q_dist
        else:
            # Standard Dueling DQN
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            
            # Combine value and advantages to get Q-values
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
            
            return q_values
            
    def get_q_dist(self, x, uncertainty=None):
        """Get the full Q-value distribution (for C51 algorithm)"""
        if not self.distributional:
            raise ValueError("get_q_dist can only be called on distributional models")
            
        q_values, q_dist = self.forward(x, uncertainty)
        return q_dist
        
    def get_q_values(self, x, uncertainty=None):
        """Get expected Q-values (mean of distribution for C51)"""
        if self.distributional:
            q_values, _ = self.forward(x, uncertainty)
            return q_values
        else:
            return self.forward(x, uncertainty)

class EnsembleDQN:
    """
    Ensemble of DQN models for uncertainty estimation.
    Uses disagreement between models as a measure of epistemic uncertainty.
    """
    def __init__(self, state_size, action_size, ensemble_size=5, distributional=False):
        self.ensemble_size = ensemble_size
        self.state_size = state_size
        self.action_size = action_size
        self.distributional = distributional
        self.models = [DuelingDQN(state_size, action_size, distributional) for _ in range(ensemble_size)]
        
    def forward(self, x):
        """Get predictions from all models in the ensemble"""
        predictions = []
        for model in self.models:
            if self.distributional:
                q_values, _ = model(x)
            else:
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