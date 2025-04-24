import pygame
import numpy as np
from src.game.terrain import Terrain, TERRAIN_COLORS
from src.game.entities import Direction

class GameRenderer:
    """Renders the game state to a pygame surface"""
    def __init__(self, window_width=1200, window_height=900, ui_height=100):
        self.window_width = window_width
        self.window_height = window_height
        self.ui_height = ui_height
        self.play_height = window_height - ui_height
        self.cell_size = 24  # Size of each grid cell in pixels
        
        # Set up pygame
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Tactical Exploration Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.large_font = pygame.font.SysFont(None, 36)
        
        # UI elements
        self.messages = []  # List of (text, expiry_time, color)
        self.danger_indicators = []  # List of (direction, expiry_time)
        
        # Sprites
        self.agent_sprites = self._create_agent_sprites()
        self.shield_sprites = self._create_shield_sprites()
        
    def _create_agent_sprites(self):
        """Create agent sprites for each direction"""
        sprites = {}
        sprite_size = 32
        
        for direction in Direction:
            sprite = pygame.Surface((sprite_size, sprite_size), pygame.SRCALPHA)
            
            # Draw agent based on direction
            body_color = (50, 100, 200)  # Blue
            head_color = (255, 220, 177)  # Flesh tone
            
            # Body (rectangle)
            pygame.draw.rect(sprite, body_color, 
                          (sprite_size//4, sprite_size//3, sprite_size//2, sprite_size//3))
            
            # Head (circle)
            head_x = sprite_size//2
            head_y = sprite_size//4
            pygame.draw.circle(sprite, head_color, (head_x, head_y), sprite_size//8)
            
            # Direction indicator
            if direction == Direction.UP:
                pygame.draw.rect(sprite, (0, 0, 0), 
                              (sprite_size//2 - 1, sprite_size//8, 2, -sprite_size//8))
            elif direction == Direction.RIGHT:
                pygame.draw.rect(sprite, (0, 0, 0), 
                              (sprite_size//2 + sprite_size//8, sprite_size//4 - 1, sprite_size//8, 2))
            elif direction == Direction.DOWN:
                pygame.draw.rect(sprite, (0, 0, 0), 
                              (sprite_size//2 - 1, sprite_size//3, 2, sprite_size//8))
            elif direction == Direction.LEFT:
                pygame.draw.rect(sprite, (0, 0, 0), 
                              (sprite_size//2 - sprite_size//4, sprite_size//4 - 1, -sprite_size//8, 2))
            
            sprites[direction] = sprite
            
        return sprites
        
    def _create_shield_sprites(self):
        """Create shield effect sprites"""
        sprites = {}
        sprite_size = 32
        
        for direction in Direction:
            sprite = pygame.Surface((sprite_size, sprite_size), pygame.SRCALPHA)
            shield_color = (100, 150, 255, 128)  # Semi-transparent blue
            
            # Draw shield as a circle around the agent
            pygame.draw.circle(sprite, shield_color, 
                            (sprite_size//2, sprite_size//2), 
                            sprite_size//2 - 2, 
                            width=2)
            
            sprites[direction] = sprite
            
        return sprites
        
    def render(self, env, agent, veto_prompt=None):
        """Render the game state"""
        # Clear the screen
        self.screen.fill((20, 20, 20))
        
        # Draw divider line between game and UI
        pygame.draw.line(self.screen, (100, 100, 100), (0, self.play_height), (self.window_width, self.play_height), 2)
        
        # Calculate camera position to center on agent
        camera_x = agent.x - self.window_width / (2 * self.cell_size)
        camera_y = agent.y - self.play_height / (2 * self.cell_size)
        
        # Clamp camera to grid boundaries
        camera_x = max(0, min(camera_x, env.grid_size - self.window_width / self.cell_size))
        camera_y = max(0, min(camera_y, env.grid_size - self.play_height / self.cell_size))
        
        # Draw visible portion of the grid
        start_x = int(camera_x)
        start_y = int(camera_y)
        end_x = min(env.grid_size, int(camera_x + self.window_width / self.cell_size) + 1)
        end_y = min(env.grid_size, int(camera_y + self.play_height / self.cell_size) + 1)
        
        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                screen_x = int((x - camera_x) * self.cell_size)
                screen_y = int((y - camera_y) * self.cell_size)
                
                # Draw terrain based on exploration status
                if 0 <= x < env.grid_size and 0 <= y < env.grid_size:
                    if env.explored[x, y]:
                        terrain = env.grid[x, y]
                        pygame.draw.rect(self.screen, TERRAIN_COLORS[terrain], (screen_x, screen_y, self.cell_size, self.cell_size))
                        
                        # Draw additional details for special terrain
                        if terrain == Terrain.HEALTH:
                            # Draw health cross
                            pygame.draw.rect(self.screen, (255, 255, 255), 
                                          (screen_x + self.cell_size//2 - 2, screen_y + 4, 4, self.cell_size - 8))
                            pygame.draw.rect(self.screen, (255, 255, 255), 
                                          (screen_x + 4, screen_y + self.cell_size//2 - 2, self.cell_size - 8, 4))
                        
                        elif terrain == Terrain.AMMO:
                            # Draw ammo icon
                            pygame.draw.rect(self.screen, (255, 255, 255), 
                                          (screen_x + self.cell_size//4, screen_y + self.cell_size//2 - 2, self.cell_size//2, 4))
                    else:
                        # Unexplored area
                        pygame.draw.rect(self.screen, (40, 40, 40), (screen_x, screen_y, self.cell_size, self.cell_size))
                
                # Light grid lines
                pygame.draw.rect(self.screen, (60, 60, 60), (screen_x, screen_y, self.cell_size, self.cell_size), 1)
        
        # Draw enemies
        for enemy in env.enemies:
            ex, ey = enemy['pos']
            if env.explored[ex, ey]:
                screen_x = int((ex - camera_x) * self.cell_size)
                screen_y = int((ey - camera_y) * self.cell_size)
                
                # Enemy type determines color
                enemy_colors = [(255, 255, 0), (255, 165, 0), (255, 70, 0)]  # Yellow, Orange, Red
                color = enemy_colors[min(enemy['type'] - 1, len(enemy_colors) - 1)]
                
                # Draw enemy body
                pygame.draw.rect(self.screen, color, (screen_x + 4, screen_y + 4, self.cell_size - 8, self.cell_size - 8))
        
        # Draw agent
        agent_screen_x = int((agent.x - camera_x) * self.cell_size)
        agent_screen_y = int((agent.y - camera_y) * self.cell_size)
        
        # Get agent sprite based on facing direction
        agent_sprite = self.agent_sprites[agent.facing]
        
        # Scale sprite to cell size if needed
        scale_factor = self.cell_size / 32
        if scale_factor != 1:
            scaled_width = int(agent_sprite.get_width() * scale_factor)
            scaled_height = int(agent_sprite.get_height() * scale_factor)
            agent_sprite = pygame.transform.scale(agent_sprite, (scaled_width, scaled_height))
        
        # Draw agent
        self.screen.blit(agent_sprite, (agent_screen_x, agent_screen_y))
        
        # Draw shield if active
        if agent.shield_active:
            shield_sprite = self.shield_sprites[agent.facing]
            if scale_factor != 1:
                shield_sprite = pygame.transform.scale(shield_sprite, (scaled_width, scaled_height))
            self.screen.blit(shield_sprite, (agent_screen_x, agent_screen_y))
        
        # Draw UI
        self._render_ui(env, agent)
        
        # Draw veto prompt if active
        if veto_prompt:
            self._render_veto_prompt(veto_prompt)
        
        pygame.display.flip()
        return self.clock.tick(60)  # Cap at 60 FPS
        
    def _render_ui(self, env, agent):
        """Render UI elements"""
        # Health bar
        health_text = self.font.render("Health:", True, (255, 255, 255))
        self.screen.blit(health_text, (20, self.play_height + 15))
        
        health_bar_width = 150
        health_bar_height = 20
        health_bar_border = pygame.Rect(20, self.play_height + 40, health_bar_width, health_bar_height)
        pygame.draw.rect(self.screen, (100, 100, 100), health_bar_border)
        
        health_percentage = agent.health / agent.max_health
        health_fill = pygame.Rect(20, self.play_height + 40, int(health_bar_width * health_percentage), health_bar_height)
        
        # Color based on health level
        if health_percentage > 0.7:
            health_color = (0, 255, 0)  # Green
        elif health_percentage > 0.3:
            health_color = (255, 255, 0)  # Yellow
        else:
            health_color = (255, 0, 0)  # Red
            
        pygame.draw.rect(self.screen, health_color, health_fill)
        
        # Health text
        health_value = self.font.render(f"{agent.health}/{agent.max_health}", True, (255, 255, 255))
        self.screen.blit(health_value, (180, self.play_height + 42))
        
        # Ammo count
        ammo_text = self.font.render(f"Ammo: {agent.ammo}", True, (255, 255, 255))
        self.screen.blit(ammo_text, (20, self.play_height + 70))
        
        # Resources
        resources_text = self.font.render(
            f"Shields: {agent.shields}   Traps: {agent.traps}   Support: {agent.support_calls}", 
            True, (255, 255, 255))
        self.screen.blit(resources_text, (300, self.play_height + 70))
        
        # Score
        score_text = self.font.render(f"Score: {agent.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, self.play_height + 100))
        
        # Enemies left
        enemies_text = self.font.render(f"Enemies: {len(env.enemies)}", True, (255, 255, 255))
        self.screen.blit(enemies_text, (300, self.play_height + 100))
        
        # Messages
        message_y = 20
        current_time = pygame.time.get_ticks() / 1000.0
        
        for i, (text, expiry_time, color) in enumerate(self.messages):
            if current_time < expiry_time:
                alpha = min(255, int(255 * (expiry_time - current_time)))
                message_surf = self.font.render(text, True, color)
                message_surf.set_alpha(alpha)
                self.screen.blit(message_surf, (self.window_width//2 - message_surf.get_width()//2, message_y))
                message_y += 25
        
        # Clear expired messages
        self.messages = [(text, expiry, color) for text, expiry, color in self.messages if current_time < expiry]
        
    def _render_veto_prompt(self, veto_prompt):
        """Render the veto prompt overlay"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.window_width, self.window_height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Prompt panel
        panel_width = min(600, self.window_width - 40)
        panel_height = min(300, self.window_height - 40)
        panel_x = (self.window_width - panel_width) // 2
        panel_y = (self.window_height - panel_height) // 2
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, (40, 40, 40), panel_rect)
        pygame.draw.rect(self.screen, (200, 50, 50), panel_rect, 2)
        
        # Title
        title_text = self.large_font.render("HIGH RISK ACTION DETECTED", True, (255, 50, 50))
        self.screen.blit(title_text, (panel_x + (panel_width - title_text.get_width()) // 2, panel_y + 20))
        
        # Action description - get from action index
        action_desc = "Unknown Action"
        if veto_prompt.action is not None:
            action_map = {
                0: "Move Up", 1: "Move Down", 2: "Move Left", 3: "Move Right",
                4: "Shoot Up", 5: "Shoot Down", 6: "Shoot Left", 7: "Shoot Right",
                8: "Reload", 9: "Take Cover", 10: "Wait"
            }
            action_desc = action_map.get(veto_prompt.action, f"Unknown Action {veto_prompt.action}")
        
        action_text = self.font.render(f"Action: {action_desc}", True, (255, 255, 255))
        self.screen.blit(action_text, (panel_x + 20, panel_y + 60))
        
        # Risk reason
        risk_text = self.font.render(f"Risk Assessment: {veto_prompt.risk_reason}", True, (255, 165, 0))
        self.screen.blit(risk_text, (panel_x + 20, panel_y + 90))
        
        # Alternatives
        if veto_prompt.alternatives:
            alt_title = self.font.render("Alternatives:", True, (200, 200, 255))
            self.screen.blit(alt_title, (panel_x + 20, panel_y + 120))
            
            for i, alt in enumerate(veto_prompt.alternatives):
                alt_text = self.font.render(f"{i+1}. {alt[1]}", True, (200, 200, 255))
                self.screen.blit(alt_text, (panel_x + 30, panel_y + 150 + i * 25))
        
        # Button: Approve
        approve_rect = pygame.Rect(panel_x + panel_width//4 - 75, panel_y + panel_height - 60, 150, 40)
        pygame.draw.rect(self.screen, (50, 150, 50), approve_rect)
        approve_text = self.font.render("Approve (Y)", True, (255, 255, 255))
        self.screen.blit(approve_text, (approve_rect.centerx - approve_text.get_width()//2, 
                                      approve_rect.centery - approve_text.get_height()//2))
        
        # Button: Veto
        veto_rect = pygame.Rect(panel_x + 3*panel_width//4 - 75, panel_y + panel_height - 60, 150, 40)
        pygame.draw.rect(self.screen, (150, 50, 50), veto_rect)
        veto_text = self.font.render("Veto (N)", True, (255, 255, 255))
        self.screen.blit(veto_text, (veto_rect.centerx - veto_text.get_width()//2, 
                                   veto_rect.centery - veto_text.get_height()//2))
        
        # Timer
        current_time = pygame.time.get_ticks() / 1000.0
        time_remaining = max(0, veto_prompt.timeout - (current_time - veto_prompt.start_time))
        timer_text = self.font.render(f"Time: {int(time_remaining)}s", True, (255, 255, 255))
        self.screen.blit(timer_text, (panel_x + panel_width - 100, panel_y + panel_height - 30))
        
    def add_message(self, text, duration=2.0, color=(255, 255, 255)):
        """Add a message to be displayed on screen"""
        current_time = pygame.time.get_ticks() / 1000.0
        self.messages.append((text, current_time + duration, color))
        
    def handle_events(self, veto_prompt=None):
        """Handle pygame events, returns (event_type, data)"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return ('QUIT', None)
                
            elif event.type == pygame.KEYDOWN:
                # Handle key presses
                if event.key == pygame.K_ESCAPE:
                    return ('QUIT', None)
                elif event.key == pygame.K_r:
                    return ('RESET', None)
                    
                # Handle veto responses
                if veto_prompt:
                    if event.key == pygame.K_y:
                        return ('VETO_RESPONSE', True)
                    elif event.key == pygame.K_n:
                        return ('VETO_RESPONSE', False)
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3]:
                        alt_index = event.key - pygame.K_1
                        if alt_index < len(veto_prompt.alternatives):
                            return ('VETO_ALTERNATIVE', alt_index)
                else:
                    # Handle movement keys
                    if event.key == pygame.K_UP:
                        return ('ACTION', 0)  # Move up
                    elif event.key == pygame.K_DOWN:
                        return ('ACTION', 1)  # Move down
                    elif event.key == pygame.K_LEFT:
                        return ('ACTION', 2)  # Move left
                    elif event.key == pygame.K_RIGHT:
                        return ('ACTION', 3)  # Move right
                    # Handle shooting keys
                    elif event.key == pygame.K_w:
                        return ('ACTION', 4)  # Shoot up
                    elif event.key == pygame.K_s:
                        return ('ACTION', 5)  # Shoot down
                    elif event.key == pygame.K_a:
                        return ('ACTION', 6)  # Shoot left
                    elif event.key == pygame.K_d:
                        return ('ACTION', 7)  # Shoot right
                    # Handle other actions
                    elif event.key == pygame.K_t:
                        return ('ACTION', 8)  # Place trap
                    elif event.key == pygame.K_c:
                        return ('ACTION', 9)  # Use cover
                    elif event.key == pygame.K_v:
                        return ('ACTION', 10)  # Call support
            
            elif event.type == pygame.MOUSEBUTTONDOWN and veto_prompt:
                # Check for clicks on veto UI buttons
                mouse_x, mouse_y = event.pos
                
                # Calculate button positions (same as in _render_veto_prompt)
                panel_width = min(600, self.window_width - 40)
                panel_height = min(300, self.window_height - 40)
                panel_x = (self.window_width - panel_width) // 2
                panel_y = (self.window_height - panel_height) // 2
                
                # Check approve button
                approve_rect = pygame.Rect(panel_x + panel_width//4 - 75, panel_y + panel_height - 60, 150, 40)
                if approve_rect.collidepoint(mouse_x, mouse_y):
                    return ('VETO_RESPONSE', True)
                
                # Check veto button
                veto_rect = pygame.Rect(panel_x + 3*panel_width//4 - 75, panel_y + panel_height - 60, 150, 40)
                if veto_rect.collidepoint(mouse_x, mouse_y):
                    return ('VETO_RESPONSE', False)
                
                # Check alternative buttons
                for i in range(min(3, len(veto_prompt.alternatives))):
                    alt_rect = pygame.Rect(panel_x + 30, panel_y + 150 + i * 25, panel_width - 60, 20)
                    if alt_rect.collidepoint(mouse_x, mouse_y):
                        return ('VETO_ALTERNATIVE', i)
                        
        return ('NONE', None)

    def resize(self, width, height):
        """Resize the rendering window"""
        self.window_width = width
        self.window_height = height
        self.screen = pygame.display.set_mode((width, height))
        # Recalculate grid cell size
        self.cell_size = min(width // 20, height // 20)  # Adjust grid to fit screen