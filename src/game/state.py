"""
State management module for the game environment.
Provides a structured way to represent and access game state.
"""

import numpy as np

class GameState:
    """
    Structured representation of game state to replace direct array access.
    This provides type safety, validation, and better organization of state components.
    """
    def __init__(self, grid_data=None, agent_stats=None, raw_state=None):
        """
        Initialize a game state object
        
        Args:
            grid_data: 2D representation of the environment grid or flattened grid data
            agent_stats: Dictionary of agent statistics (health, ammo, shields)
            raw_state: Raw state array (alternative to providing components separately)
        """
        self.grid_size = None
        self.grid_data = None
        self.agent_stats = {
            'health': 1.0,
            'ammo': 1.0,
            'shields': 1.0
        }
        self.raw = None
        
        # Initialize from raw state if provided
        if raw_state is not None:
            self._init_from_raw(raw_state)
        elif grid_data is not None:
            self._init_from_components(grid_data, agent_stats)
        else:
            raise ValueError("Either grid_data or raw_state must be provided")
    
    def _init_from_raw(self, raw_state):
        """Initialize from raw state array"""
        if not isinstance(raw_state, (np.ndarray, list)):
            raise ValueError("Raw state must be a numpy array or list")
            
        raw_state = np.array(raw_state)
        self.raw = raw_state
        
        # Extract agent stats (assuming last 3 elements are stats)
        self.agent_stats['health'] = float(raw_state[-3])
        self.agent_stats['ammo'] = float(raw_state[-2])
        self.agent_stats['shields'] = float(raw_state[-1])
        
        # Extract grid data
        grid_data_extracted = raw_state[:-3]
        self.grid_size = int(np.sqrt(len(grid_data_extracted)))
        
        # Reshape grid data if dimensions work out
        if self.grid_size**2 == len(grid_data_extracted):
            self.grid_data = grid_data_extracted.reshape(self.grid_size, self.grid_size)
        else:
            # Can't reshape, keep as flat array
            self.grid_data = grid_data_extracted
            
    def _init_from_components(self, grid_data, agent_stats):
        """Initialize from separate components"""
        if grid_data is None:
            raise ValueError("grid_data cannot be None when initializing from components")
        # Process grid data
        if isinstance(grid_data, np.ndarray):
            if grid_data.ndim == 2:
                # 2D grid
                self.grid_size = grid_data.shape[0]
                self.grid_data = grid_data
            else:
                # Flattened grid
                self.grid_data = grid_data
                # Try to determine grid size if it's a square
                size = int(np.sqrt(len(grid_data)))
                if size**2 == len(grid_data):
                    self.grid_size = size
                    self.grid_data = grid_data.reshape(size, size)
        else:
            raise ValueError("Grid data must be a numpy array")
            
        # Process agent stats
        if agent_stats is not None:
            for key, value in agent_stats.items():
                if key in self.agent_stats:
                    self.agent_stats[key] = float(value)
                    
        # Create raw representation
        self._update_raw()
    
    def _update_raw(self):
        """Update the raw state representation from components"""
        # Flatten grid data if it's 2D
        flat_grid = self.grid_data.flatten() if self.grid_data.ndim == 2 else self.grid_data
        
        # Concatenate with agent stats
        self.raw = np.concatenate([
            flat_grid,
            np.array([
                self.agent_stats['health'],
                self.agent_stats['ammo'],
                self.agent_stats['shields']
            ])
        ])
    
    def __array__(self):
        """Allow numpy array conversion"""
        return self.raw
        
    def to_tensor(self):
        """Convert to PyTorch tensor for model input"""
        import torch
        return torch.FloatTensor(self.raw)
        
    # Accessor methods for agent stats
    @property
    def health(self):
        return self.agent_stats['health']
        
    @health.setter
    def health(self, value):
        self.agent_stats['health'] = float(value)
        self._update_raw()
        
    @property
    def ammo(self):
        return self.agent_stats['ammo']
        
    @ammo.setter
    def ammo(self, value):
        self.agent_stats['ammo'] = float(value)
        self._update_raw()
        
    @property
    def shields(self):
        return self.agent_stats['shields']
        
    @shields.setter
    def shields(self, value):
        self.agent_stats['shields'] = float(value)
        self._update_raw()
        
    def is_low_health(self, threshold=0.3):
        """Check if health is below threshold"""
        return self.health < threshold
        
    def is_low_ammo(self, threshold=0.2):
        """Check if ammo is below threshold"""
        return self.ammo < threshold
        
    def has_shields(self):
        """Check if agent has shields"""
        return self.shields > 0
        
    def get_size(self):
        """Get the total size of the state vector"""
        return len(self.raw)
        
    def validate(self):
        """Validate state values are within expected ranges"""
        # Check agent stats are in [0,1] range
        for stat, value in self.agent_stats.items():
            if not 0 <= value <= 1:
                return False, f"Agent stat '{stat}' out of range: {value}"
        
        return True, "State is valid"