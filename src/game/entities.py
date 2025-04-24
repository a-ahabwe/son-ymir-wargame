from enum import IntEnum

class Direction(IntEnum):
    """Direction constants"""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Entity:
    """Base class for game entities"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    @property
    def pos(self):
        return (self.x, self.y)

class Agent(Entity):
    """Player agent"""
    def __init__(self, x, y):
        super().__init__(x, y)
        self.health = 100
        self.max_health = 100
        self.ammo = 15
        self.shields = 3
        self.shield_active = False
        self.score = 0
        self.facing = Direction.DOWN
        
        # Resources and abilities
        self.traps = 3
        self.support_calls = 1
        
        # State flags
        self.in_cover = False
        
        # Stats for metrics
        self.shots_taken = 0
        self.shots_hit = 0
        self.traps_placed = 0
        self.traps_triggered = 0
        self.cover_used = 0
        self.support_calls_used = 0
        
    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        
        # Update facing direction
        if dx > 0:
            self.facing = Direction.RIGHT
        elif dx < 0:
            self.facing = Direction.LEFT
        elif dy > 0:
            self.facing = Direction.DOWN
        elif dy < 0:
            self.facing = Direction.UP
            
    def take_damage(self, amount):
        """Agent takes damage, returns True if agent died"""
        # Check if shield is active
        if self.shield_active:
            # Shields reduce damage by 75%
            reduced_amount = amount * 0.25
            
            # Reduce shields count and deactivate if at 0
            self.shields -= 1
            if self.shields <= 0:
                self.shield_active = False
                
            # Apply reduced damage
            self.health = max(0, self.health - reduced_amount)
        else:
            # Take full damage if no shield
            self.health = max(0, self.health - amount)
            
        return self.health <= 0
        
    def toggle_shield(self):
        """Toggle shield on/off if available"""
        if self.shields <= 0:
            return False, "No shields left"
            
        self.shield_active = not self.shield_active
        return True, "Shield activated" if self.shield_active else "Shield deactivated"
        
    def place_trap(self):
        """Place a trap at current position"""
        if self.traps <= 0:
            return False, "No traps left"
            
        self.traps -= 1
        self.traps_placed += 1
        return True, "Trap placed"
        
    def use_cover(self, cover_quality):
        """Use cover at current position"""
        if cover_quality <= 0:
            return False, "No cover available"
            
        self.in_cover = not self.in_cover
        
        if self.in_cover:
            self.cover_used += 1
            return True, f"Taking cover (quality: {cover_quality:.2f})"
        else:
            return True, "Leaving cover"
            
    def call_support(self):
        """Call for support strike"""
        if self.support_calls <= 0:
            return False, "No support calls left"
            
        self.support_calls -= 1
        self.support_calls_used += 1
        return True, "Called for support strike"

class Enemy(Entity):
    """Enemy entity"""
    def __init__(self, x, y, enemy_type=1):
        super().__init__(x, y)
        self.type = enemy_type  # 1=low, 2=medium, 3=high value
        self.health = 20 * enemy_type
        self.facing = Direction.DOWN
        
    def get_value(self):
        """Get point value of this enemy"""
        return self.type * 25  # Low=25, Medium=50, High=75