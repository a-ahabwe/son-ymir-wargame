"""
Adapter module to integrate the GameState class with existing code.
Provides backward compatibility while transitioning to the new state model.
"""

import numpy as np
from src.game.state import GameState

class StateAdapter:
    """
    Adapter to convert between raw state arrays and GameState objects.
    Provides backward compatibility while transitioning to structured state.
    """
    
    @staticmethod
    def from_environment(env):
        """
        Create a GameState from the environment
        
        Args:
            env: GameEnvironment instance
            
        Returns:
            GameState object
        """
        # Get the raw state array from environment
        raw_state = env._get_state()
        
        # Create and return a GameState
        return GameState(raw_state=raw_state)
    
    @staticmethod
    def to_raw(game_state):
        """
        Convert GameState to raw state array (for backward compatibility)
        
        Args:
            game_state: GameState object
            
        Returns:
            numpy array representation
        """
        return game_state.raw
    
    @staticmethod
    def adapt_environment_step(env, action):
        """
        Wrap environment step to return GameState instead of raw array
        
        Args:
            env: GameEnvironment instance
            action: Action to take
            
        Returns:
            (GameState, reward, done, info) tuple
        """
        next_state, reward, done, info = env.step(action)
        return GameState(raw_state=next_state), reward, done, info
    
    @staticmethod
    def adapt_for_model(state):
        """
        Adapt state for model input, handling both GameState and raw arrays
        
        Args:
            state: GameState object or raw state array
            
        Returns:
            Raw state array suitable for model input
        """
        if isinstance(state, GameState):
            return state.raw
        return state
        
    @classmethod
    def update_environment(cls, env):
        """
        Update environment to use GameState by monkey patching
        
        Args:
            env: GameEnvironment instance
            
        Returns:
            Modified environment
        """
        # Store original _get_state method
        original_get_state = env._get_state
        
        # Create wrapper for _get_state to return GameState
        def get_state_wrapper():
            raw_state = original_get_state()
            return GameState(raw_state=raw_state)
            
        # Replace the method
        env._get_state = get_state_wrapper
        
        # Store original step method
        original_step = env.step
        
        # Create wrapper for step to return GameState
        def step_wrapper(action):
            next_state, reward, done, info = original_step(action)
            return GameState(raw_state=next_state), reward, done, info
            
        # Replace the method
        env.step = step_wrapper
        
        return env