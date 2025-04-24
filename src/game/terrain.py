import numpy as np
import random
import math
from enum import IntEnum

class Terrain(IntEnum):
    """Terrain types for the game environment"""
    GRASS = 1
    TREE = 2
    WATER = 3
    MOUNTAIN = 4
    MINE = 5
    HEALTH = 6
    AMMO = 7
    SHIELD = 8
    SWAMP = 9
    HILL = 10
    FOREST = 11
    SAND = 12

# Color mapping for visualization
TERRAIN_COLORS = {
    Terrain.GRASS: (128, 204, 51),      # Light green
    Terrain.TREE: (51, 128, 51),        # Dark green
    Terrain.WATER: (51, 102, 204),      # Blue
    Terrain.MOUNTAIN: (153, 153, 153),  # Gray
    Terrain.MINE: (204, 25, 25),        # Red
    Terrain.HEALTH: (25, 230, 76),      # Bright green
    Terrain.SAND: (230, 210, 150),      # Tan
    Terrain.AMMO: (255, 165, 0),        # Orange
    Terrain.SHIELD: (100, 150, 255),    # Light blue
    Terrain.SWAMP: (110, 90, 70),       # Brown
    Terrain.HILL: (180, 180, 100),      # Light brown
    Terrain.FOREST: (70, 150, 70)       # Medium green
}

# Terrain effects
TERRAIN_EFFECTS = {
    Terrain.GRASS: {"movement_cost": 1.0, "cover": 0.0, "description": "grass"},
    Terrain.TREE: {"movement_cost": float('inf'), "cover": 0.0, "description": "tree"},
    Terrain.WATER: {"movement_cost": 3.0, "cover": 0.0, "description": "water"},
    Terrain.MOUNTAIN: {"movement_cost": float('inf'), "cover": 0.0, "description": "mountain"},
    Terrain.MINE: {"movement_cost": 1.0, "cover": 0.0, "description": "mine"},
    Terrain.HEALTH: {"movement_cost": 1.0, "cover": 0.0, "description": "health pack"},
    Terrain.SAND: {"movement_cost": 1.5, "cover": 0.0, "description": "sand"},
    Terrain.AMMO: {"movement_cost": 1.0, "cover": 0.0, "description": "ammo pack"},
    Terrain.SHIELD: {"movement_cost": 1.0, "cover": 0.0, "description": "shield"},
    Terrain.SWAMP: {"movement_cost": 2.0, "cover": 0.0, "description": "swamp"},
    Terrain.HILL: {"movement_cost": 1.5, "cover": 0.3, "description": "hill"},
    Terrain.FOREST: {"movement_cost": 1.3, "cover": 0.5, "description": "forest"}
}

class TerrainGenerator:
    """Class to handle procedural terrain generation"""
    def __init__(self, grid_size, seed=None):
        self.grid_size = grid_size
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def generate_terrain_grid(self):
        """Generate a grid with natural-looking terrain"""
        grid = np.ones((self.grid_size, self.grid_size), dtype=int) * Terrain.GRASS
        
        # Generate noise maps for terrain
        elevation = self._generate_noise_map(scale=20.0)
        moisture = self._generate_noise_map(scale=15.0)
        
        # Apply terrain based on elevation and moisture
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                elev = elevation[x, y]
                moist = moisture[x, y]
                
                if elev < 0.2:
                    if moist > 0.6:
                        grid[x, y] = Terrain.WATER
                    elif moist > 0.3:
                        grid[x, y] = Terrain.SWAMP
                    else:
                        grid[x, y] = Terrain.SAND
                elif elev < 0.5:
                    if moist > 0.7:
                        grid[x, y] = Terrain.FOREST
                    else:
                        grid[x, y] = Terrain.GRASS
                elif elev < 0.7:
                    # Medium elevations get trees, hills or grass
                    if moist > 0.6:
                        # Ensure there are paths through forests
                        if (x + y) % 3 == 0:
                            grid[x, y] = Terrain.GRASS
                        else:
                            grid[x, y] = Terrain.TREE
                    elif moist > 0.3:
                        grid[x, y] = Terrain.HILL
                    else:
                        grid[x, y] = Terrain.GRASS
                else:
                    # Create paths through mountain ranges
                    if (x % 4 == 0 or y % 4 == 0):
                        grid[x, y] = Terrain.HILL
                    else:
                        grid[x, y] = Terrain.MOUNTAIN
        
        return grid
    
    def _generate_noise_map(self, scale=10.0):
        """Generate a simple noise map"""
        noise = np.zeros((self.grid_size, self.grid_size))
        
        # Simplified noise generation - in a real implementation,
        # you might want to use Perlin noise or similar
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                nx = x / scale
                ny = y / scale
                # Use some trigonometric functions to create a noise-like pattern
                noise[x, y] = (
                    math.sin(nx) * math.cos(ny) + 
                    math.sin(nx * 2) * math.cos(ny * 2) * 0.5 + 
                    math.sin(nx * 4) * math.cos(ny * 4) * 0.25
                )
                # Normalize to 0-1 range
                noise[x, y] = (noise[x, y] + 2) / 4
                
        return noise
    
    def ensure_reachability(self, grid, start_x, start_y):
        """Make sure all passable cells are reachable from start position"""
        # Here you would implement a pathfinding algorithm to check reachability
        # and modify the grid to create paths if needed
        # This is a placeholder implementation
        passable = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                terrain = grid[x, y]
                effects = TERRAIN_EFFECTS.get(terrain, {"movement_cost": float('inf')})
                passable[x, y] = effects["movement_cost"] < float('inf')
        
        # Run a simple flood fill to identify reachable areas
        reachable = np.zeros_like(passable)
        self._flood_fill(passable, reachable, start_x, start_y)
        
        # Create paths to unreachable areas (simplified implementation)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if passable[x, y] and not reachable[x, y]:
                    # Create a path to this unreachable area
                    self._create_path(grid, start_x, start_y, x, y)
        
        return grid
    
    def _flood_fill(self, passable, reachable, x, y):
        """Flood fill algorithm to mark reachable cells"""
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return
        if not passable[x, y] or reachable[x, y]:
            return
            
        reachable[x, y] = True
        
        # Recursively check adjacent cells
        self._flood_fill(passable, reachable, x+1, y)
        self._flood_fill(passable, reachable, x-1, y)
        self._flood_fill(passable, reachable, x, y+1)
        self._flood_fill(passable, reachable, x, y-1)
    
    def _create_path(self, grid, start_x, start_y, target_x, target_y):
        """Create a path between two points by modifying terrain"""
        # Simple implementation - draw a straight line with grass
        dx = abs(target_x - start_x)
        dy = abs(target_y - start_y)
        sx = 1 if start_x < target_x else -1
        sy = 1 if start_y < target_y else -1
        err = dx - dy
        
        x, y = start_x, start_y
        
        while x != target_x or y != target_y:
            grid[x, y] = Terrain.GRASS  # Set current cell to grass
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
                
        grid[target_x, target_y] = Terrain.GRASS  # Ensure destination is grass