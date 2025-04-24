#!/usr/bin/env python3
"""
Experiment runner for the Veto Game project.
Runs the experiment with specified parameters and conditions.
"""

import argparse
import os
import time
import random
import sys
from datetime import datetime
from pathlib import Path

# Ensure the src package is in the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.game.environment import GameEnvironment
from src.game.main import Game
from src.experiment.scenarios import ScenarioGenerator
from src.experiment.data_collection import ExperimentLogger
from src.ai.uncertainty import UncertaintyEstimator
from src.ai.agent import RLAgent
from src.veto.veto_mechanism import ThresholdVetoMechanism, UncertaintyVetoMechanism

def create_experiment_id():
    """Create a unique experiment ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"experiment_{timestamp}"

def setup_experiment(experiment_id, participants, conditions):
    """Set up the experiment directory and configuration"""
    # Create experiment directory
    base_dir = f"data/experiment_results/{experiment_id}"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create logger
    logger = ExperimentLogger(base_dir)
    
    # Generate scenarios
    scenario_generator = ScenarioGenerator(seed=42)
    scenarios = scenario_generator.generate_scenarios(count=5)
    
    # Save experiment configuration
    config = {
        "experiment_id": experiment_id,
        "participants": participants,
        "conditions": conditions,
        "scenarios": [s.name for s in scenarios],
        "start_time": time.time()
    }
    
    return config, logger, scenarios

def run_experiment_for_participant(participant_id, conditions, scenarios, experiment_id, headless=False):
    """Run the experiment for a single participant across all conditions"""
    print(f"Running experiment for participant {participant_id}")
    
    # Create base environment
    env = GameEnvironment(grid_size=50, seed=42)
    
    # Create RL agent
    agent = RLAgent(env.state_size, env.action_space_n)
    
    # Create uncertainty estimator
    uncertainty_estimator = UncertaintyEstimator(agent.model)
    
    # Set up data collection
    logger = ExperimentLogger(f"data/experiment_results/{experiment_id}")
    
    # Randomize condition order for counterbalancing
    condition_order = list(conditions)
    random.shuffle(condition_order)
    
    for condition in condition_order:
        # Start session
        logger.start_session(participant_id, condition)
        print(f"  Condition: {condition}")
        
        # Create appropriate veto mechanism OR set to None for baseline
        veto_mechanism = None
        if condition == "threshold":
            veto_mechanism = ThresholdVetoMechanism(timeout=10)
        elif condition == "uncertainty":
            veto_mechanism = UncertaintyVetoMechanism(uncertainty_estimator, timeout=10)
        elif condition == 'baseline':
            pass # No veto mechanism needed for baseline
        else:
            raise ValueError(f"Unknown condition: {condition}")
        
        # Run each scenario
        for scenario in scenarios:
            print(f"    Scenario: {scenario.name}")
            
            # Apply scenario to environment
            current_env = scenario.apply_to_environment(env)
            
            # Create game instance
            game = Game(
                env=current_env,
                headless=headless,
                session_id=f"{participant_id}_{condition}_{scenario.name}",
                log_dir=f"data/experiment_results/{experiment_id}/logs",
                condition=condition,
                uncertainty_threshold=0.7,
                logger=logger
            )
            game.veto = veto_mechanism # Explicitly set the veto mechanism on the game instance
            
            # Run game for this scenario
            # In headless mode, run for fixed number of steps
            if headless:
                max_steps = 200
                
                # Pass the logger to the game instance
                game.logger = logger
                
                state = game.reset()
                for step in range(max_steps):
                    # Select action using AI
                    action, q_values = game.rl_agent.select_action(state)
                    final_action = action
                    veto_decision = None # Initialize veto_decision
                    vetoed = False       # Default to not vetoed

                    # Check for veto only if condition is not baseline and veto mechanism exists
                    if condition != 'baseline' and game.veto:
                        veto_decision = game.veto.assess_action(state, action, q_values)

                        if veto_decision.vetoed:
                            # In headless mode, simulate random veto decision
                            vetoed = random.random() < 0.3  # 30% chance of veto

                            if vetoed:
                                # Select alternative action (simple random for headless)
                                alternative_action = random.randint(0, game.env.action_space_n - 1)
                                final_action = alternative_action
                            else:
                                alternative_action = None # Veto assessed but not applied

                            # Log veto decision (only if veto occurred)
                            if game.logger:
                                # Use ExperimentLogger directly if available
                                if hasattr(game, 'data_collector') and game.data_collector:
                                    game.data_collector.log_veto({
                                        'original_action': action,
                                        'vetoed': vetoed,
                                        'alternative': alternative_action,
                                        'simulated': True,
                                        'reason': veto_decision.reason,
                                        'risk_reason': veto_decision.risk_reason,
                                        'uncertainty': veto_decision.uncertainty,
                                        'threshold': veto_decision.threshold
                                    })
                                # Fallback to the basic logger if needed
                                else:
                                    game.logger.log_veto({
                                        'original_action': action,
                                        'vetoed': vetoed,
                                        'alternative': alternative_action,
                                        'simulated': True
                                    })

                    # Execute the final action in the environment
                    next_state, reward, done, info = game.env.step(final_action)

                    # Log the executed action and outcome using DataCollector
                    if hasattr(game, 'data_collector') and game.data_collector:
                        game.data_collector.log_action(
                            action=final_action,
                            state=state, # Log the state *before* the action
                            next_state=next_state,
                            reward=reward,
                            done=done,
                            info=info,
                            q_values=q_values.tolist() if q_values is not None else None, # Convert numpy array
                            veto_applied=veto_decision.vetoed if veto_decision else False
                        )

                    # Update state for the next iteration
                    state = next_state
                    
                    if done:
                        break
            else:
                # In windowed mode, run normal game loop
                game.run()
        
        # End session
        logger.end_session()
    
    print(f"Finished experiment for participant {participant_id}")

def main():
    parser = argparse.ArgumentParser(description='Run experiment for Veto Game')
    parser.add_argument('--participants', type=int, default=1, help='Number of participants')
    parser.add_argument('--conditions', nargs='+', default=['baseline', 'threshold', 'uncertainty'],
                      help='Conditions to test (baseline, threshold, uncertainty)')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--experiment_id', type=str, default=None, help='Experiment ID (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Validate conditions
    valid_conditions = {'baseline', 'threshold', 'uncertainty'}
    for condition in args.conditions:
        if condition not in valid_conditions:
            print(f"Error: Unknown condition '{condition}'. Valid conditions are: {', '.join(valid_conditions)}")
            return
    
    # Create experiment ID if not provided
    experiment_id = args.experiment_id or create_experiment_id()
    
    print(f"Starting experiment {experiment_id} with {args.participants} participants")
    print(f"Conditions: {', '.join(args.conditions)}")
    
    # Set up experiment
    config, logger, scenarios = setup_experiment(experiment_id, args.participants, args.conditions)
    
    # Run experiment for each participant
    for participant_id in range(1, args.participants + 1):
        run_experiment_for_participant(
            participant_id, 
            args.conditions, 
            scenarios, 
            experiment_id,
            headless=args.headless
        )
    
    print(f"Experiment {experiment_id} completed!")
    
    # Record experiment end time
    config["end_time"] = time.time()
    config["duration"] = config["end_time"] - config["start_time"]
    
    # Save final config
    import json
    with open(f"data/experiment_results/{experiment_id}/experiment_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Results saved to data/experiment_results/{experiment_id}/")
    print("To analyze results, run:")
    print(f"  python -m src.experiment.analyze --data_dir data/experiment_results/{experiment_id}")

if __name__ == "__main__":
    main()