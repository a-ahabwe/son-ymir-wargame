"""
Dataset module for veto decisions.
Provides tools to create, load, and manage labeled veto datasets.
"""

import numpy as np
import os
import json
import random
import pickle
import time
from collections import defaultdict
from src.game.state import GameState
from src.veto.ground_truth import GroundTruthOracle

class VetoDataset:
    """
    Dataset of labeled veto decisions.
    Provides a set of state-action pairs with ground truth veto labels.
    """
    def __init__(self, name="veto_dataset"):
        """
        Initialize a veto dataset
        
        Args:
            name: Name of the dataset
        """
        self.name = name
        self.data = []
        self.metadata = {
            'name': name,
            'created': time.time(),
            'version': '1.0',
            'size': 0,
            'balanced': False,
            'features': {
                'state_size': None,
                'action_space_n': None
            },
            'statistics': {
                'veto_count': 0,
                'noveto_count': 0,
                'veto_ratio': 0.0
            }
        }
        
    def add_example(self, state, action, should_veto, q_values=None, confidence=1.0, explanation=""):
        """
        Add a labeled example to the dataset
        
        Args:
            state: GameState object or raw state array
            action: Action that was taken
            should_veto: Whether this action should be vetoed (ground truth)
            q_values: Optional Q-values for the state
            confidence: Confidence in the ground truth label (0.0-1.0)
            explanation: Optional explanation for the label
        """
        # Convert state to GameState if it's not already
        if not isinstance(state, GameState):
            state = GameState(raw_state=state)
        
        # Extract features for more efficient storage
        example = {
            'state_array': state.raw,
            'state_stats': {
                'health': state.health,
                'ammo': state.ammo,
                'shields': state.shields
            },
            'action': action,
            'should_veto': should_veto,
            'q_values': q_values.tolist() if q_values is not None and hasattr(q_values, 'tolist') else q_values,
            'confidence': confidence,
            'explanation': explanation
        }
        
        # Update metadata
        if self.metadata['features']['state_size'] is None:
            self.metadata['features']['state_size'] = len(state.raw)
            
        if self.metadata['features']['action_space_n'] is None and q_values is not None:
            self.metadata['features']['action_space_n'] = len(q_values)
            
        # Add to dataset
        self.data.append(example)
        
        # Update statistics
        if should_veto:
            self.metadata['statistics']['veto_count'] += 1
        else:
            self.metadata['statistics']['noveto_count'] += 1
            
        self.metadata['size'] = len(self.data)
        self.metadata['statistics']['veto_ratio'] = (
            self.metadata['statistics']['veto_count'] / max(1, self.metadata['size'])
        )
            
    def balance_dataset(self, target_ratio=0.5, method='undersample'):
        """
        Balance the dataset to have a specified ratio of veto/no-veto examples
        
        Args:
            target_ratio: Target ratio of veto examples (0.0-1.0)
            method: Balancing method ('undersample', 'oversample', or 'hybrid')
            
        Returns:
            Number of examples after balancing
        """
        veto_examples = [ex for ex in self.data if ex['should_veto']]
        noveto_examples = [ex for ex in self.data if not ex['should_veto']]
        
        veto_count = len(veto_examples)
        noveto_count = len(noveto_examples)
        
        if veto_count == 0 or noveto_count == 0:
            # Can't balance if one class is empty
            return len(self.data)
            
        # Calculate target counts
        total = veto_count + noveto_count
        target_veto = int(total * target_ratio)
        target_noveto = total - target_veto
        
        balanced_data = []
        
        if method == 'undersample':
            # Undersample the majority class
            if veto_count > target_veto:
                # Too many veto examples
                veto_examples = random.sample(veto_examples, target_veto)
            else:
                # Too many no-veto examples
                noveto_examples = random.sample(noveto_examples, total - target_veto)
                
            balanced_data = veto_examples + noveto_examples
            
        elif method == 'oversample':
            # Oversample the minority class
            if veto_count < target_veto:
                # Need more veto examples
                additional = target_veto - veto_count
                veto_examples += random.choices(veto_examples, k=additional)
            else:
                # Need more no-veto examples
                additional = target_noveto - noveto_count
                noveto_examples += random.choices(noveto_examples, k=additional)
                
            balanced_data = veto_examples + noveto_examples
            
        elif method == 'hybrid':
            # Hybrid approach: slight undersampling of majority and oversampling of minority
            if veto_count > noveto_count:
                # More veto examples
                target_veto = min(veto_count, int(1.5 * noveto_count))
                veto_examples = random.sample(veto_examples, target_veto)
                target_noveto = noveto_count
            else:
                # More no-veto examples
                target_noveto = min(noveto_count, int(1.5 * veto_count))
                noveto_examples = random.sample(noveto_examples, target_noveto)
                target_veto = veto_count
                
            balanced_data = veto_examples + noveto_examples
            
        # Update dataset
        self.data = balanced_data
        
        # Update metadata
        self.metadata['balanced'] = True
        self.metadata['size'] = len(self.data)
        self.metadata['statistics']['veto_count'] = len([ex for ex in self.data if ex['should_veto']])
        self.metadata['statistics']['noveto_count'] = len([ex for ex in self.data if not ex['should_veto']])
        self.metadata['statistics']['veto_ratio'] = (
            self.metadata['statistics']['veto_count'] / max(1, self.metadata['size'])
        )
        
        return len(self.data)
        
    def split_train_test(self, test_ratio=0.2, stratify=True):
        """
        Split dataset into training and test sets
        
        Args:
            test_ratio: Ratio of examples to use for testing
            stratify: Whether to maintain class distribution in splits
            
        Returns:
            (train_dataset, test_dataset) tuple
        """
        # Create new datasets
        train_dataset = VetoDataset(f"{self.name}_train")
        test_dataset = VetoDataset(f"{self.name}_test")
        
        if stratify:
            # Stratified split (maintain veto ratio)
            veto_examples = [ex for ex in self.data if ex['should_veto']]
            noveto_examples = [ex for ex in self.data if not ex['should_veto']]
            
            # Calculate split sizes
            veto_test_size = int(len(veto_examples) * test_ratio)
            noveto_test_size = int(len(noveto_examples) * test_ratio)
            
            # Shuffle
            random.shuffle(veto_examples)
            random.shuffle(noveto_examples)
            
            # Split
            veto_test = veto_examples[:veto_test_size]
            veto_train = veto_examples[veto_test_size:]
            noveto_test = noveto_examples[:noveto_test_size]
            noveto_train = noveto_examples[noveto_test_size:]
            
            # Add to datasets
            for ex in veto_train + noveto_train:
                train_dataset.add_example(
                    ex['state_array'],
                    ex['action'],
                    ex['should_veto'],
                    ex['q_values'],
                    ex['confidence'],
                    ex['explanation']
                )
                
            for ex in veto_test + noveto_test:
                test_dataset.add_example(
                    ex['state_array'],
                    ex['action'],
                    ex['should_veto'],
                    ex['q_values'],
                    ex['confidence'],
                    ex['explanation']
                )
        else:
            # Random split
            test_size = int(len(self.data) * test_ratio)
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            
            test_indices = indices[:test_size]
            train_indices = indices[test_size:]
            
            # Add to datasets
            for i in train_indices:
                ex = self.data[i]
                train_dataset.add_example(
                    ex['state_array'],
                    ex['action'],
                    ex['should_veto'],
                    ex['q_values'],
                    ex['confidence'],
                    ex['explanation']
                )
                
            for i in test_indices:
                ex = self.data[i]
                test_dataset.add_example(
                    ex['state_array'],
                    ex['action'],
                    ex['should_veto'],
                    ex['q_values'],
                    ex['confidence'],
                    ex['explanation']
                )
        
        return train_dataset, test_dataset
        
    def get_batch(self, batch_size=32, balanced=True):
        """
        Get a batch of examples
        
        Args:
            batch_size: Size of batch to return
            balanced: Whether to balance veto/no-veto examples in batch
            
        Returns:
            list of examples
        """
        if balanced:
            # Get approximately equal numbers of each class
            veto_examples = [ex for ex in self.data if ex['should_veto']]
            noveto_examples = [ex for ex in self.data if not ex['should_veto']]
            
            # Calculate how many of each to include
            veto_count = min(len(veto_examples), batch_size // 2)
            noveto_count = min(len(noveto_examples), batch_size - veto_count)
            
            # Adjust if we don't have enough of one class
            if veto_count < batch_size // 2:
                noveto_count = min(len(noveto_examples), batch_size - veto_count)
                
            # Sample examples
            batch_veto = random.sample(veto_examples, veto_count)
            batch_noveto = random.sample(noveto_examples, noveto_count)
            
            # Combine and shuffle
            batch = batch_veto + batch_noveto
            random.shuffle(batch)
            
            return batch
        else:
            # Random batch
            if batch_size >= len(self.data):
                return self.data.copy()
            
            return random.sample(self.data, batch_size)
            
    def save(self, directory, compress=True):
        """
        Save dataset to file
        
        Args:
            directory: Directory to save in
            compress: Whether to compress the data
            
        Returns:
            Path to saved files
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save metadata as JSON
        metadata_path = os.path.join(directory, f"{self.name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save data
        if compress:
            data_path = os.path.join(directory, f"{self.name}_data.pkl.gz")
            with open(data_path, 'wb') as f:
                pickle.dump(self.data, f)
        else:
            data_path = os.path.join(directory, f"{self.name}_data.json")
            with open(data_path, 'w') as f:
                json.dump(self.data, f, indent=2, default=lambda o: o.tolist() 
                                                              if isinstance(o, np.ndarray) else o)
        
        return metadata_path, data_path
    
    @classmethod
    def load(cls, directory, name):
        """
        Load dataset from file
        
        Args:
            directory: Directory to load from
            name: Name of dataset
            
        Returns:
            Loaded VetoDataset
        """
        # Create new instance
        dataset = cls(name)
        
        # Load metadata
        metadata_path = os.path.join(directory, f"{name}_metadata.json")
        with open(metadata_path, 'r') as f:
            dataset.metadata = json.load(f)
        
        # Try to load compressed data first
        compressed_path = os.path.join(directory, f"{name}_data.pkl.gz")
        if os.path.exists(compressed_path):
            with open(compressed_path, 'rb') as f:
                dataset.data = pickle.load(f)
        else:
            # Fall back to JSON
            data_path = os.path.join(directory, f"{name}_data.json")
            with open(data_path, 'r') as f:
                dataset.data = json.load(f)
        
        return dataset


class VetoDatasetGenerator:
    """
    Generator for creating veto datasets using ground truth.
    """
    def __init__(self, environment_class, oracle=None):
        """
        Initialize dataset generator
        
        Args:
            environment_class: Class of the environment (not instance)
            oracle: Optional pre-configured GroundTruthOracle
        """
        self.environment_class = environment_class
        self.oracle = oracle or GroundTruthOracle(environment_class)
        
    def generate_dataset(self, agent, size=1000, balanced=True, 
                        env_params=None, name="veto_dataset"):
        """
        Generate a labeled dataset of veto decisions
        
        Args:
            agent: Agent to use for generating actions
            size: Number of examples to generate
            balanced: Whether to balance veto/no-veto examples
            env_params: Optional parameters for environment
            name: Name for the dataset
            
        Returns:
            Generated VetoDataset
        """
        # Create dataset
        dataset = VetoDataset(name)
        
        # Create environment
        env_params = env_params or {}
        env = self.environment_class(**env_params)
        
        # Track statistics for status updates
        examples_collected = 0
        veto_count = 0
        no_veto_count = 0
        target_veto_ratio = 0.5 if balanced else None
        
        print(f"Generating dataset with {size} examples...")
        
        # Generate examples
        while examples_collected < size:
            # Reset environment for a new episode
            state = env.reset()
            done = False
            
            # Run an episode
            while not done and examples_collected < size:
                # Select action
                action, q_values = agent.select_action(state)
                
                # Determine if this action should be vetoed (ground truth)
                should_veto, confidence, explanation = self.oracle.should_veto(state, action, q_values)
                
                # Check if we should add this example (for balanced dataset)
                add_example = True
                if balanced:
                    # Current ratio of veto examples
                    current_ratio = veto_count / max(1, veto_count + no_veto_count)
                    
                    if should_veto and current_ratio >= target_veto_ratio:
                        # Skip if we already have enough veto examples
                        if no_veto_count < size / 2:  # Unless we need more examples
                            add_example = False
                    elif not should_veto and current_ratio < target_veto_ratio:
                        # Skip if we already have enough no-veto examples
                        if veto_count < size / 2:  # Unless we need more examples
                            add_example = False
                
                # Add example if appropriate
                if add_example:
                    dataset.add_example(state, action, should_veto, q_values, confidence, explanation)
                    examples_collected += 1
                    
                    if should_veto:
                        veto_count += 1
                    else:
                        no_veto_count += 1
                        
                    # Print progress
                    if examples_collected % 100 == 0:
                        print(f"  Generated {examples_collected}/{size} examples "
                             f"({veto_count} veto, {no_veto_count} no-veto)")
                
                # Take action in environment
                next_state, reward, done, _ = env.step(action)
                state = next_state
                
                # Break if done
                if done:
                    break
        
        # Balance dataset if needed
        if balanced and abs(dataset.metadata['statistics']['veto_ratio'] - target_veto_ratio) > 0.05:
            print("Final dataset balancing...")
            dataset.balance_dataset(target_ratio=target_veto_ratio)
        
        print(f"Dataset generation complete: {len(dataset.data)} examples "
             f"({dataset.metadata['statistics']['veto_count']} veto, "
             f"{dataset.metadata['statistics']['noveto_count']} no-veto)")
        
        return dataset