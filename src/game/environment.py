import numpy as np
import random
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

class GameEnvironment:
    """Core game environment independent of rendering or control logic"""
    def __init__(self, grid_size=100, seed=None):
        self.grid_size = grid_size
        self.seed = seed
        self.grid = None
        self.agent_pos = None
        self.enemies = []
        self.agent_health = 100
        self.agent_ammo = 15
        self.agent_shields = 3
        self.explored = None
        self.reset()

        # Define action space
        self.actions = {
            0: self._move_up,     # Move up
            1: self._move_down,   # Move down
            2: self._move_left,   # Move left
            3: self._move_right,  # Move right
            4: self._shoot_up,    # Shoot up
            5: self._shoot_down,  # Shoot down
            6: self._shoot_left,  # Shoot left
            7: self._shoot_right, # Shoot right
            8: self._place_trap,  # Place trap
            9: self._use_cover,   # Use cover
            10: self._call_support  # Request support
        }
        self.action_space_n = len(self.actions)
        
        # State dimensions
        # Flattened grid + agent stats (health, ammo, shields)
        self.state_size = grid_size * grid_size + 3

    def reset(self):
        """Reset the environment and return initial state"""
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        # Generate grid
        self.grid = self._generate_terrain()
        
        # Place agent
        self.agent_pos = (self.grid_size // 2, self.grid_size // 2)
        self.grid[self.agent_pos] = Terrain.GRASS  # Ensure agent starts on grass
        
        # Reset agent stats
        self.agent_health = 100
        self.agent_ammo = 15
        self.agent_shields = 3
        
        # Generate paths to ensure reachability
        self._ensure_reachability()
        
        # Place objects
        self._place_objects()
        
        # Initialize explored area
        self.explored = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self._update_exploration(initial=True)
        
        return self._get_state()

    def step(self, action):
        """Execute an action and return new state, reward, done, info"""
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}. Must be between 0 and {self.action_space_n-1}")
        
        # Execute action
        success, reward, info = self.actions[action]()
        
        # Update exploration
        self._update_exploration()
        
        # Update enemies
        self._update_enemies()
        
        # Check if game is over
        done = self.agent_health <= 0 or not self.enemies
        
        # Return state, reward, done, info
        return self._get_state(), reward, done, info

    def _generate_terrain(self):
        """Generate a procedural terrain map"""
        # Simple implementation - replace with more sophisticated generator
        grid = np.ones((self.grid_size, self.grid_size), dtype=int) * Terrain.GRASS
        
        # Add some obstacles and terrain features
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                r = np.random.random()
                if r < 0.02:
                    grid[x, y] = Terrain.TREE
                elif r < 0.04:
                    grid[x, y] = Terrain.WATER
                elif r < 0.06:
                    grid[x, y] = Terrain.MOUNTAIN
                elif r < 0.08:
                    grid[x, y] = Terrain.SWAMP
                elif r < 0.10:
                    grid[x, y] = Terrain.HILL
                elif r < 0.12:
                    grid[x, y] = Terrain.FOREST
                elif r < 0.14:
                    grid[x, y] = Terrain.SAND
        
        print("Terrain generated with features:", np.unique(grid, return_counts=True))
        return grid

    def _ensure_reachability(self):
        """Make sure all passable cells are reachable"""
        # Simple implementation - make a path to key areas
        # In a full implementation, use a pathfinding algorithm to check reachability
        pass

    def _place_objects(self):
        """Place objects (mines, health, ammo, enemies) on the grid"""
        # Place mines
        mine_positions = self._place_items(Terrain.MINE, 10)
        print("Mines placed at:", mine_positions)
        
        # Place health packs
        health_positions = self._place_items(Terrain.HEALTH, 15)
        print("Health packs placed at:", health_positions)
        
        # Place ammo packs
        ammo_positions = self._place_items(Terrain.AMMO, 12)
        print("Ammo packs placed at:", ammo_positions)
        
        # Place shield pickups
        shield_positions = self._place_items(Terrain.SHIELD, 5)
        print("Shields placed at:", shield_positions)
        
        # Place enemies
        enemy_positions = self._place_items(Terrain.MOUNTAIN, 20, temporary=True)
        self.enemies = []
        for x, y in enemy_positions:
            self.grid[x, y] = Terrain.GRASS  # Clear temporary marker
            self.enemies.append({
                'pos': (x, y),
                'health': 20,
                'type': np.random.randint(1, 4)  # 1=low, 2=medium, 3=high value
            })
        print("Enemies placed at:", [enemy['pos'] for enemy in self.enemies])

    def _place_items(self, item_type, count, temporary=False):
        """Place items on the grid, returning their positions"""
        positions = []
        for _ in range(count):
            for _ in range(100):  # Try up to 100 times to find a valid location
                x, y = np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)
                # Check if location is grass and not the agent's position
                if self.grid[x, y] == Terrain.GRASS and (x, y) != self.agent_pos:
                    self.grid[x, y] = item_type
                    positions.append((x, y))
                    break
        return positions

    def _update_exploration(self, initial=False):
        """Update the explored area around the agent"""
        vision_radius = 12 if initial else 8
        x, y = self.agent_pos
        
        for dx in range(-vision_radius, vision_radius + 1):
            for dy in range(-vision_radius, vision_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance <= vision_radius and self._has_line_of_sight(x, y, nx, ny):
                        self.explored[nx, ny] = True

    def _has_line_of_sight(self, x1, y1, x2, y2):
        """Check if there is a clear line of sight between two points"""
        # Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while x1 != x2 or y1 != y2:
            if (self.grid[x1, y1] == Terrain.TREE or 
                self.grid[x1, y1] == Terrain.MOUNTAIN):
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
                
        return True

    def _update_enemies(self):
        """Update enemy behavior"""
        # Simple implementation
        # In a full game, implement more sophisticated AI
        pass

    def _get_state(self):
        """Return current state representation"""
        # Create a flattened representation of the grid
        # 0 = unexplored, positive = terrain type, negative = enemy
        state_grid = np.zeros((self.grid_size, self.grid_size))
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.explored[x, y]:
                    state_grid[x, y] = self.grid[x, y]
                    
                    # Mark enemy positions
                    for i, enemy in enumerate(self.enemies):
                        if enemy['pos'] == (x, y):
                            state_grid[x, y] = -enemy['type']  # Negative indicates enemy
        
        # Flatten grid and add agent stats
        flattened = state_grid.flatten()
        agent_state = np.array([
            self.agent_health / 100.0,  # Normalize to [0,1]
            self.agent_ammo / 30.0,     # Normalize
            self.agent_shields / 5.0    # Normalize
        ])
        
        return np.concatenate([flattened, agent_state])

    # Action methods
    def _move_up(self):
        x, y = self.agent_pos
        return self._move(x, y-1)
        
    def _move_down(self):
        x, y = self.agent_pos
        return self._move(x, y+1)
        
    def _move_left(self):
        x, y = self.agent_pos
        return self._move(x-1, y)
        
    def _move_right(self):
        x, y = self.agent_pos
        return self._move(x+1, y)

    def _move(self, new_x, new_y):
        """Move the agent to a new position"""
        # Check if the move is valid
        if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
            return False, -0.1, {"message": "Cannot move out of bounds"}
            
        terrain = self.grid[new_x, new_y]
        effects = TERRAIN_EFFECTS.get(terrain, {"movement_cost": float('inf')})
        
        # Check if terrain is passable
        if effects["movement_cost"] == float('inf'):
            return False, -0.1, {"message": f"Cannot move to {effects['description']}"}
            
        # Move agent
        self.agent_pos = (new_x, new_y)
        
        # Handle terrain effects
        reward = 0.1  # Base reward for movement
        message = f"Moved to {effects['description']}"
        
        # Handle special terrain
        if terrain == Terrain.MINE:
            self.agent_health -= 15
            self.grid[new_x, new_y] = Terrain.GRASS  # Mine is triggered
            reward += 5 if self.agent_health > 0 else -10  # Reward for surviving
            message = "Hit a mine! -15 health"
            
        elif terrain == Terrain.HEALTH:
            old_health = self.agent_health
            self.agent_health = min(100, self.agent_health + 20)
            self.grid[new_x, new_y] = Terrain.GRASS
            reward += 10
            message = f"Health pack! +{self.agent_health - old_health} health"
            
        elif terrain == Terrain.AMMO:
            self.agent_ammo += 10
            self.grid[new_x, new_y] = Terrain.GRASS
            reward += 5
            message = "Ammo pack! +10 ammo"
            
        elif terrain == Terrain.SHIELD:
            self.agent_shields += 1
            self.grid[new_x, new_y] = Terrain.GRASS
            reward += 10
            message = f"Shield pickup! Now have {self.agent_shields} shields"
            
        return True, reward, {"message": message}

    def _shoot_up(self):
        return self._shoot(0, -1)
        
    def _shoot_down(self):
        return self._shoot(0, 1)
        
    def _shoot_left(self):
        return self._shoot(-1, 0)
        
    def _shoot_right(self):
        return self._shoot(1, 0)
        
    def _shoot(self, dx, dy):
        """Shoot in a direction"""
        if self.agent_ammo <= 0:
            return False, -0.2, {"message": "No ammunition left"}
            
        self.agent_ammo -= 1
        
        # Check what the shot hits
        hit = False
        hit_pos = None
        hit_type = None
        max_range = 5
        
        x, y = self.agent_pos
        for i in range(1, max_range + 1):
            nx, ny = x + dx * i, y + dy * i
            
            # Check if out of bounds
            if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                break
                
            # Check if hit obstacle
            terrain = self.grid[nx, ny]
            if terrain in [Terrain.TREE, Terrain.MOUNTAIN]:
                hit = True
                hit_pos = (nx, ny)
                hit_type = terrain
                break
                
            # Check if hit enemy
            for i, enemy in enumerate(self.enemies):
                if enemy['pos'] == (nx, ny):
                    hit = True
                    hit_pos = (nx, ny)
                    hit_type = "enemy"
                    
                    # Remove enemy
                    enemy_value = enemy['type'] * 25  # Low=25, Medium=50, High=75
                    self.enemies.pop(i)
                    
                    # Add reward based on enemy value
                    return True, enemy_value / 25.0, {
                        "message": f"Enemy eliminated! +{enemy_value} points",
                        "hit": hit,
                        "hit_pos": hit_pos,
                        "hit_type": hit_type
                    }
        
        if hit:
            return True, 0.2, {
                "message": f"Shot hit {hit_type} at {hit_pos}", 
                "hit": hit, 
                "hit_pos": hit_pos, 
                "hit_type": hit_type
            }
        else:
            return True, 0, {
                "message": "Shot missed", 
                "hit": False
            }

    def _place_trap(self):
        """Place a trap at the current position"""
        # Simplified implementation
        return True, 0.2, {"message": "Trap placed"}
        
    def _use_cover(self):
        """Use cover if available"""
        x, y = self.agent_pos
        terrain = self.grid[x, y]
        effects = TERRAIN_EFFECTS.get(terrain, {"cover": 0.0})
        
        if effects["cover"] > 0:
            return True, 0.3, {"message": f"Using cover (quality: {effects['cover']:.2f})"}
        else:
            return False, 0, {"message": "No cover available on current terrain"}
            
    def _call_support(self):
        """Call for support strike"""
        # Find nearest enemy within range
        max_range = 5
        x, y = self.agent_pos
        
        nearest_enemy = None
        nearest_dist = float('inf')
        
        for i, enemy in enumerate(self.enemies):
            ex, ey = enemy['pos']
            dist = abs(ex - x) + abs(ey - y)
            if dist <= max_range and dist < nearest_dist:
                nearest_enemy = i
                nearest_dist = dist
                
        if nearest_enemy is None:
            return False, -0.1, {"message": "No targets within range"}
            
        # 75% chance to hit
        if np.random.random() < 0.75:
            enemy = self.enemies[nearest_enemy]
            ex, ey = enemy['pos']
            enemy_value = enemy['type'] * 37.5  # 1.5x normal value
            
            # Remove enemy
            self.enemies.pop(nearest_enemy)
            
            return True, enemy_value / 25.0, {
                "message": f"Support strike hit enemy! +{enemy_value} points",
                "hit_pos": (ex, ey)
            }
        else:
            return True, 0, {"message": "Support strike missed"}