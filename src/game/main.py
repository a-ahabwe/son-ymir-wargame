import pygame
import time
import argparse
import numpy as np
from src.game.environment import GameEnvironment
from src.game.entities import Agent, Direction
from src.game.rendering import GameRenderer
from src.ai.agent import RLAgent
from src.veto.veto_mechanism import VetoMechanism, ThresholdVetoMechanism, UncertaintyVetoMechanism
from src.experiment.data_collection import ExperimentLogger
from src.ai.uncertainty import UncertaintyEstimator
from src.experiment.data_collector import DataCollector
import uuid
import logging

class VetoPrompt:
    """Represents a veto decision prompt"""
    def __init__(self, state=None, action=None, reason="", risk_reason="", q_values=None, timeout=10):
        self.state = state
        self.action = action
        self.reason = reason
        self.risk_reason = risk_reason
        self.q_values = q_values
        self.start_time = time.time()
        self.timeout = timeout
        self.response = None
        self.alternatives = []
        self.selected_alternative = None
        
    def set_alternatives(self, alternatives):
        """Set alternative actions"""
        self.alternatives = alternatives

class Game:
    """Main game class that integrates environment, agent, and veto mechanism"""
    def __init__(self, env=None, headless=False, session_id=None, log_dir=None, condition=None, 
                 uncertainty_threshold=0.7, enable_distributional=True, enable_attention=True):
        """
        Initialize the Game
        
        Args:
            env: Optional environment to use
            headless: Whether to run in headless mode
            session_id: Session ID for data collection
            log_dir: Directory to store logs
            condition: Experiment condition ('uncertainty', 'threshold', or None)
            uncertainty_threshold: Threshold for veto decisions
            enable_distributional: Whether to use distributional RL
            enable_attention: Whether to use attention mechanisms
        """
        self.headless = headless
        self.session_id = session_id or str(uuid.uuid4())
        self.log_dir = log_dir
        self.condition = condition
        self.uncertainty_threshold = uncertainty_threshold
        self.training_mode = True  # Whether to train the RL agent during play
        self.veto_enabled = True   # Re-enable veto mechanism but with adjusted settings
        self.veto_events = []      # Store veto events
        self.interface_state = "normal"  # UI state
        self.waiting_for_input = False   # Whether waiting for user input
        self.verbose = False  # Enable verbose logging
        
        # Logger setup
        self.logger = self._setup_logger()
        
        # Environment setup
        self.env = env or GameEnvironment()
        
        # AI agent setup
        self.rl_agent = RLAgent(
            self.env.state_size, 
            self.env.action_space_n,
            distributional=enable_distributional,
            uncertainty_driven=True,
            num_atoms=51,
            v_min=-10,
            v_max=10
        )
        
        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(
            self.rl_agent.model, 
            self.env.action_space_n,
            distributional=enable_distributional,
            mc_dropout_samples=10
        )
        
        # Game entities
        self.player = None
        
        # Initialize veto mechanism
        self._initialize_veto(condition, uncertainty_threshold)
        
        # Game state
        self.reset()
        
        # Data collection
        if log_dir:
            self.data_collector = DataCollector(session_id, log_dir)
        else:
            self.data_collector = None
        
        # Stats
        self.stats = {
            'total_reward': 0,
            'veto_requests': 0,
            'veto_approvals': 0,
            'veto_rejections': 0,
            'veto_timeouts': 0
        }
        
        # Initialize renderer (only in windowed mode)
        if not headless:
            self.renderer = GameRenderer(1200, 900)
        
        # Game state
        self.running = True
        self.veto_prompt = None
        self.last_action_time = 0
        self.action_delay = 0.2  # Seconds between AI actions (balance between too fast and too slow)
        
    def _setup_logger(self):
        """Initialize logging system"""
        logger = logging.getLogger('game')
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handler
        logger.addHandler(ch)
        
        # File handler if log directory provided
        if self.log_dir:
            import os
            os.makedirs(self.log_dir, exist_ok=True)
            fh = logging.FileHandler(f"{self.log_dir}/game_{self.session_id}.log")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
        return logger
        
    def _initialize_veto(self, condition, uncertainty_threshold):
        """Initialize the appropriate veto mechanism based on condition"""
        if condition == 'uncertainty':
            self.veto = UncertaintyVetoMechanism(
                uncertainty_estimator=self.uncertainty_estimator,
                uncertainty_threshold=uncertainty_threshold,
                timeout=10,
                use_adaptive_threshold=True
            )
        elif condition == 'threshold':
            self.veto = ThresholdVetoMechanism(
                threshold=0.7,  # Default threshold for risk assessment
                timeout=10
            )
        else:
            # Default veto mechanism with uncertainty-based approach
            self.veto = UncertaintyVetoMechanism(
                uncertainty_estimator=self.uncertainty_estimator,
                uncertainty_threshold=uncertainty_threshold,
                timeout=10,
                use_adaptive_threshold=True
            )
        
    def reset(self):
        """Reset the game"""
        state = self.env.reset()
        x, y = self.env.agent_pos
        self.player = Agent(x, y)
        return state
        
    def run(self):
        """Run the game loop"""
        state = self.reset()
        self.running = True
        
        try:
            while self.running:
                # Handle pygame events
                event_type, event_data = self.renderer.handle_events(self.veto_prompt)
                
                if event_type == 'QUIT':
                    self.running = False
                    break
                    
                elif event_type == 'RESET':
                    state = self.reset()
                    continue
                    
                elif event_type == 'VETO_RESPONSE':
                    if self.veto_prompt:
                        self.veto_prompt.response = event_data
                        if not event_data:  # Vetoed
                            if self.veto_prompt.selected_alternative is not None:
                                # Execute alternative action
                                action = self.veto_prompt.selected_alternative
                                self._execute_action(action, state)
                                # Log veto decision if experiment is active
                                if self.data_collector:
                                    self.data_collector.log_veto({
                                        'original_action': self.veto_prompt.action,
                                        'vetoed': True,
                                        'alternative': action
                                    })
                            else:
                                # No alternative selected, don't do anything
                                if self.data_collector:
                                    self.data_collector.log_veto({
                                        'original_action': self.veto_prompt.action,
                                        'vetoed': True,
                                        'alternative': None
                                    })
                        else:
                            # Approved, execute original action
                            action = self.veto_prompt.action
                            self._execute_action(action, state)
                            # Log veto decision
                            if self.data_collector:
                                self.data_collector.log_veto({
                                    'original_action': action,
                                    'vetoed': False,
                                    'alternative': None
                                })
                        
                        self.veto_prompt = None
                        
                elif event_type == 'VETO_ALTERNATIVE':
                    if self.veto_prompt and event_data < len(self.veto_prompt.alternatives):
                        alt_action = self.veto_prompt.alternatives[event_data][0]
                        self.veto_prompt.selected_alternative = alt_action
                        self.veto_prompt.response = False  # Vetoed
                        
                elif event_type == 'ACTION':
                    # Manual action from user
                    action = event_data
                    
                    # Check if this action needs veto
                    veto_decision = self.veto.assess_action(state, action, None)
                    
                    if veto_decision.vetoed:
                        # Create veto prompt
                        action_desc = self._get_action_description(action)
                        self.veto_prompt = VetoPrompt(
                            state=state, 
                            action=action, 
                            reason=veto_decision.reason, 
                            risk_reason=veto_decision.risk_reason, 
                            q_values=None, 
                            timeout=self.veto.timeout
                        )
                        
                        # Get alternative actions
                        alternatives = self._get_alternative_actions(action)
                        self.veto_prompt.set_alternatives(alternatives)
                        
                        # Log veto request
                        if self.data_collector:
                            self.data_collector.log_veto_request({
                                'action': action,
                                'action_desc': action_desc,
                                'risk_reason': veto_decision.risk_reason
                            })
                    else:
                        # Execute action without veto
                        self._execute_action(action, state)
                
                # AI action if no veto prompt is active
                current_time = time.time()
                if not self.veto_prompt and current_time - self.last_action_time >= self.action_delay:
                    self.last_action_time = current_time
                    
                    # Select action using RL agent
                    action, q_values = self.rl_agent.select_action(state, self.uncertainty_estimator)
                    
                    # Check if this action needs veto
                    veto_decision = self.veto.assess_action(state, action, q_values)
                    
                    if veto_decision.vetoed:
                        # Create veto prompt
                        action_desc = self._get_action_description(action)
                        self.veto_prompt = VetoPrompt(
                            state=state, 
                            action=action, 
                            reason=veto_decision.reason, 
                            risk_reason=veto_decision.risk_reason, 
                            q_values=q_values, 
                            timeout=self.veto.timeout
                        )
                        
                        # Get alternative actions
                        alternatives = self._get_alternative_actions(action)
                        self.veto_prompt.set_alternatives(alternatives)
                        
                        # Log veto request
                        if self.data_collector:
                            self.data_collector.log_veto_request({
                                'action': action,
                                'action_desc': action_desc,
                                'risk_reason': veto_decision.risk_reason
                            })
                    else:
                        # Execute action without veto
                        self._execute_action(action, state)
                
                # Check for veto timeout
                if self.veto_prompt:
                    current_time = time.time()
                    if current_time - self.veto_prompt.start_time >= self.veto_prompt.timeout:
                        # Default to executing original action on timeout
                        action = self.veto_prompt.action
                        self._execute_action(action, state)
                        
                        # Log timeout
                        if self.data_collector:
                            self.data_collector.log_veto({
                                'original_action': action,
                                'vetoed': False,
                                'alternative': None,
                                'timeout': True
                            })
                        
                        self.veto_prompt = None
                        self.renderer.add_message("Veto timeout - proceeding with action", 2.0, (255, 165, 0))
                
                # Render
                self.renderer.render(self.env, self.player, self.veto_prompt)
                
                # Check if game is over
                if self.env.agent_health <= 0:
                    self.renderer.add_message("Game Over! Health reached zero.", 5.0, (255, 0, 0))
                    pygame.time.wait(2000)  # Wait 2 seconds
                    state = self.reset()
                    
                elif not self.env.enemies:
                    self.renderer.add_message("Victory! All enemies eliminated.", 5.0, (0, 255, 0))
                    pygame.time.wait(2000)  # Wait 2 seconds
                    state = self.reset()
            
            # ... rest of the method ...
        finally:
            # Save data when game is closed
            if self.data_collector:
                self.data_collector.save()
                print("Session data saved.")
        
    def _execute_action(self, action, state=None, uncertainty=None):
        """
        Execute an action with uncertainty awareness
        
        Args:
            action: The action to execute
            state: Current state (will retrieve if None)
            uncertainty: Uncertainty estimation if available
        """
        # Get current state if not provided
        if state is None:
            state = self.env.get_state()
            
        # Execute action in environment
        next_state, reward, done, info = self.env.step(action)
        
        # Update player position from environment
        if hasattr(self.env, 'agent_pos'):
            x, y = self.env.agent_pos
            # Update player object
            if self.player:
                self.player.x = x
                self.player.y = y
        
        # Update total reward
        if hasattr(self, 'stats'):
            self.stats['total_reward'] += reward
        
        # Log the action and reward, including uncertainty if available
        action_info = {
            'action': action,
            'reward': reward,
            'q_value': self.rl_agent.get_q_value(state, action) if hasattr(self.rl_agent, 'get_q_value') else None,
        }
        
        if uncertainty is not None:
            action_info['uncertainty'] = uncertainty
            
        if hasattr(self, 'data_collector') and self.data_collector:
            self.data_collector.log_action(**action_info)
        
        # Store transition for training
        if hasattr(self, 'training_mode') and self.training_mode:
            self.rl_agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent if we have enough samples in replay buffer
            if len(self.rl_agent.replay_buffer) > self.rl_agent.batch_size:
                self.rl_agent.train()
                
        return reward, done
    
    def _get_action_description(self, action):
        """Get a human-readable description of an action"""
        action_map = {
            0: "Move Up",
            1: "Move Down",
            2: "Move Left",
            3: "Move Right",
            4: "Shoot Up",
            5: "Shoot Down",
            6: "Shoot Left",
            7: "Shoot Right",
            8: "Reload",
            9: "Take Cover",
            10: "Wait"
        }
        return action_map.get(action, f"Unknown Action {action}")
    
    def _get_alternative_actions(self, current_action, num_alternatives=3):
        """Get alternative actions based on current state"""
        # Get current state
        state = self.env._get_state()
        
        # Get Q-values for current state
        q_values = self.rl_agent.get_q_values(state)
        
        # Mask the current action
        q_values_np = q_values.copy()
        q_values_np[current_action] = float('-inf')
        
        # Find highest Q-value alternatives
        alternatives = []
        for _ in range(min(num_alternatives, self.env.action_space_n - 1)):
            alt_action = np.argmax(q_values_np)
            alt_q_value = q_values[alt_action]
            
            # Get action description
            alt_action_desc = self._get_action_description(alt_action)
            
            alternatives.append((alt_action, alt_action_desc, alt_q_value))
            
            # Mask this action for the next iteration
            q_values_np[alt_action] = float('-inf')
            
        return alternatives

    def run_ai_turn(self):
        """
        Run a single AI agent turn with veto mechanism
        
        Returns:
            Whether the game is still running
        """
        current_state = self.env.get_state()
        
        # Get action from AI agent
        action, q_values = self.rl_agent.select_action(current_state, explore=self.training_mode)
        original_action = action  # Store original action for learning
        uncertainty = None
        veto_applied = False
        
        # Assess action for veto if enabled
        if self.veto_enabled and self.veto:
            # Get uncertainty estimation if available
            if hasattr(self.uncertainty_estimator, 'estimate_uncertainty'):
                uncertainty = self.uncertainty_estimator.estimate_uncertainty(current_state, action)
            elif hasattr(self.uncertainty_estimator, 'decision_uncertainty'):
                uncertainty = self.uncertainty_estimator.decision_uncertainty(current_state, action)
            
            veto_decision = self.veto.assess_action(
                state=current_state,
                action=action,
                q_values=q_values,
                uncertainty=uncertainty
            )
            
            if veto_decision.vetoed:
                # If action was vetoed, get alternative action using our safe action selector
                veto_applied = True
                alternative_action = self.rl_agent.select_safe_action(
                    current_state, 
                    current_action=action,
                    uncertainty_estimator=self.uncertainty_estimator
                )
                action = alternative_action or action  # Fallback to original action if no alternative
                veto_decision.alternative_action = action  # Record the alternative chosen
                self.log_veto_event(veto_decision)
        
        # Execute the chosen action
        result = self._execute_action(action, state=current_state, uncertainty=uncertainty)
        reward, done = result
        
        # Learn from veto experience
        if veto_applied and hasattr(self.rl_agent, 'learn_from_veto'):
            self.rl_agent.learn_from_veto(current_state, original_action, action, reward)
            
        # If veto decision was recorded, update it with outcome
        if veto_applied and self.veto:
            self.veto.record_veto_decision(
                state=current_state,
                action=original_action,
                vetoed=True,
                alternative=action,
                outcome=result
            )
        
        # Check if game is over
        if done:
            self.rl_agent.end_episode(self.env.get_state(), self.env.get_reward())
            
        return not done
    
    def log_veto_event(self, veto_decision):
        """
        Log a veto event with detailed information
        
        Args:
            veto_decision: VetoDecision object containing veto information
        """
        if not hasattr(self, 'veto_events'):
            self.veto_events = []
            
        # Record the veto event
        veto_info = {
            'original_action': veto_decision.original_action,
            'alternative_action': veto_decision.alternative_action,
            'reason': veto_decision.reason,
            'uncertainty': veto_decision.uncertainty,
            'threshold': veto_decision.threshold if hasattr(veto_decision, 'threshold') else None,
            'q_values': veto_decision.q_values.tolist() if hasattr(veto_decision.q_values, 'tolist') else veto_decision.q_values,
            'timestamp': time.time()
        }
        
        self.veto_events.append(veto_info)
        
        # Print veto event to console
        print(f"VETO EVENT: Action {veto_decision.original_action} vetoed. Reason: {veto_decision.reason}")
        print(f"VETO DETAILS: Uncertainty={veto_decision.uncertainty}, Threshold={veto_decision.threshold}")
        
        # Log to data collector if available
        if hasattr(self, 'data_collector') and self.data_collector:
            self.data_collector.log_veto_event(**veto_info)
            
        # Write to debug log file directly
        try:
            import os
            import json
            debug_log_dir = "data/logs/debug"
            os.makedirs(debug_log_dir, exist_ok=True)
            debug_log_file = f"{debug_log_dir}/veto_events_{self.session_id}.json"
            
            # Load existing data if file exists
            events = []
            if os.path.exists(debug_log_file):
                try:
                    with open(debug_log_file, 'r') as f:
                        events = json.load(f)
                except:
                    events = []
            
            # Add new event
            events.append(veto_info)
            
            # Write to file
            with open(debug_log_file, 'w') as f:
                json.dump(events, f, indent=2)
            
            print(f"VETO EVENT saved to {debug_log_file}")
        except Exception as e:
            print(f"Error writing veto event to debug log: {str(e)}")
        
        # Print info to console if in verbose mode
        if self.verbose:
            print(f"VETO: Action {veto_decision.original_action} vetoed. Reason: {veto_decision.reason}")
            print(f"Alternative action selected: {veto_decision.alternative_action}")

    def ai_turn(self):
        """Execute AI agent's turn"""
        if not self.rl_agent:
            self.logger.warning("AI agent not initialized")
            return
            
        self.logger.info("AI turn started")
        state = self.env.get_state()
        
        # Get action from AI agent
        action, q_values = self.rl_agent.select_action(state, explore=self.training_mode)
        original_action = action
        
        # Check if veto mechanism is active
        if self.veto:
            self.logger.info(f"AI selected action: {self._get_action_description(action)}")
            
            # Assess the action using veto mechanism
            veto_decision = self.veto.assess_action(state, action, q_values)
            
            # If action is vetoed
            if veto_decision.vetoed:
                self.logger.warning(f"VETO triggered: {veto_decision.reason}")
                self.logger.info(f"Risk details: {veto_decision.risk_reason}")
                
                # Get alternative actions
                alternative_actions = self._get_alternative_actions(action)
                
                if self.headless:
                    # Autonomous mode - let AI choose alternative
                    if alternative_actions and len(alternative_actions) > 0:
                        action = alternative_actions[0][0]
                        self.logger.info(f"AI selected alternative action: {self._get_action_description(action)}")
                        veto_decision.alternative_action = action
                    else:
                        self.logger.warning("No alternative actions available, proceeding with original")
                        # Keep original action
                else:
                    # Interactive mode - ask human operator
                    # Create veto prompt
                    self.veto_prompt = VetoPrompt(
                        state=state, 
                        action=action, 
                        reason=veto_decision.reason, 
                        risk_reason=veto_decision.risk_reason, 
                        q_values=veto_decision.q_values, 
                        timeout=self.veto.timeout
                    )
                    
                    # Get alternative actions
                    self.veto_prompt.set_alternatives(alternative_actions)
                    
                    # Set UI state
                    self.interface_state = "veto_prompt"
                    self.waiting_for_input = True
                    
                    # Wait for veto decision
                    while self.waiting_for_input:
                        time.sleep(0.1)
                        # Check for timeout
                        if time.time() - self.veto_prompt.start_time > self.veto_prompt.timeout:
                            self.logger.warning("Veto timeout - proceeding with original action")
                            self.waiting_for_input = False
                    
                    # Process veto response
                    if self.veto_prompt.selected_alternative is not None:
                        action = self.veto_prompt.selected_alternative
                        self.logger.info(f"Human selected alternative action: {self._get_action_description(action)}")
                        veto_decision.alternative_action = action
        
        # Execute action and get result
        result = self._execute_action(action, state=state)
        
        # Record the veto decision outcome
        if self.veto and veto_decision.vetoed:
            self.veto.record_veto_decision(
                state=state,
                action=original_action,
                vetoed=True,
                alternative=action if action != original_action else None,
                outcome=result
            )
            
        self.logger.info(f"AI action executed: {self._get_action_description(action)}")
        
        # Process result
        self._process_turn_result(result)
        self.logger.info("AI turn completed")

    def _process_turn_result(self, result):
        """Process the result of a turn"""
        reward, done = result
        
        # Update game state based on result
        if done:
            self.logger.info(f"Game over! Final reward: {reward}")
            # Trigger game over state
            
        # Update UI if needed
        if not self.headless:
            # Update UI with result
            pass

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Tactical Exploration Game with AI and Veto')
    parser.add_argument('--grid_size', type=int, default=100, help='Size of the game grid')
    parser.add_argument('--window_width', type=int, default=1200, help='Window width in pixels')
    parser.add_argument('--window_height', type=int, default=900, help='Window height in pixels')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--veto_mechanism', choices=['threshold', 'uncertainty'], default='threshold',
                      help='Veto mechanism to use')
    parser.add_argument('--veto_timeout', type=int, default=10, help='Timeout for veto decisions in seconds')
    parser.add_argument('--experiment_id', type=str, default=None, help='Experiment ID for logging')
    parser.add_argument('--uncertainty_threshold', type=float, default=0.3, help='Threshold for uncertainty-based veto')
    
    args = parser.parse_args()
    
    # Create environment with grid size
    env = GameEnvironment(grid_size=args.grid_size)
    
    # Create game with appropriate parameters
    game = Game(
        env=env,
        headless=False,
        session_id=args.experiment_id,
        log_dir='data/logs',
        condition=args.veto_mechanism,
        uncertainty_threshold=args.uncertainty_threshold  # Use command line argument
    )
    
    # Set veto timeout
    if game.veto:
        game.veto.timeout = args.veto_timeout
    
    # Set renderer dimensions if created
    if hasattr(game, 'renderer'):
        game.renderer.resize(args.window_width, args.window_height)
    
    game.run()

if __name__ == "__main__":
    main()