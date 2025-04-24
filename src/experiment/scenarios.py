import random
import numpy as np

class TestScenario:
    """Defines a standardized test scenario for consistent evaluation"""
    def __init__(self, name, difficulty=0.5, seed=None):
        self.name = name
        self.difficulty = difficulty  # 0.0 to 1.0
        self.seed = seed
        self.params = self._generate_params()
    
    def _generate_params(self):
        """Generate scenario parameters based on difficulty"""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            
        # Basic parameters that scale with difficulty
        params = {
            'grid_size': 50 + int(self.difficulty * 50),  # 50-100
            'enemy_count': 5 + int(self.difficulty * 25),  # 5-30
            'mine_count': int(self.difficulty * 20),  # 0-20
            'health_count': 20 - int(self.difficulty * 10),  # 20-10
            'ammo_count': 15 - int(self.difficulty * 5),  # 15-10
            'initial_ammo': 20 - int(self.difficulty * 10),  # 20-10
            'enemy_aggression': 0.2 + self.difficulty * 0.6,  # 0.2-0.8
            'enemy_types': {
                'low': 1.0 - self.difficulty,  # Decreases with difficulty
                'medium': 0.5,
                'high': 0.0 + self.difficulty  # Increases with difficulty
            }
        }
        return params
    
    def apply_to_environment(self, env):
        """Apply scenario parameters to environment"""
        # Set random seeds
        if self.seed is not None:
            env.seed = self.seed
            random.seed(self.seed)
            np.random.seed(self.seed)
            
        # Modify environment parameters
        # This is a simplified version - in a real implementation,
        # you would modify the actual environment parameters
        env.grid_size = self.params['grid_size']
        
        # Return modified environment
        return env

class ScenarioGenerator:
    """Generates a set of test scenarios with varying difficulties"""
    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
    def generate_scenarios(self, count=5):
        """Generate a set of test scenarios"""
        scenarios = []
        
        # Generate evenly spaced difficulties
        difficulties = np.linspace(0.1, 0.9, count)
        
        # Create named scenarios
        names = [
            "Tutorial",
            "Easy",
            "Medium",
            "Hard",
            "Extreme"
        ]
        
        # Generate scenarios
        for i in range(count):
            if i < len(names):
                name = names[i]
            else:
                name = f"Scenario {i+1}"
                
            # Use deterministic seeds derived from the master seed
            scenario_seed = None
            if self.seed is not None:
                scenario_seed = self.seed + i
                
            scenario = TestScenario(name, difficulties[i], scenario_seed)
            scenarios.append(scenario)
            
        return scenarios