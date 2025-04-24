"""
Learning-based risk assessor that improves from experience.
Replaces hand-crafted heuristics with a supervised learning approach.
"""

import numpy as np
import pandas as pd
import time
import os
import joblib
from collections import deque

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LearningRiskAssessor:
    """
    Risk assessor that learns from experience.
    Uses machine learning to classify risky actions.
    """
    def __init__(self, model_path=None, model_type="random_forest"):
        # Model and pipeline
        self.model = None
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        # Experience buffer for training
        self.experience_buffer = deque(maxlen=10000)
        
        # Performance metrics
        self.metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "sample_count": 0
        }
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._create_model()
        
        # Feature cache to avoid repeated extraction
        self.feature_cache = {}
        self.cache_timeout = 5.0  # seconds
        
    def _create_model(self):
        """Create a new machine learning model"""
        if self.model_type == "random_forest":
            base_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            base_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == "neural_network":
            base_model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
        else:
            # Default to RandomForest
            base_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            
        # Create pipeline with scaling
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', base_model)
        ])
        
    def extract_features(self, state, action, q_values=None):
        """
        Extract features for risk assessment
        
        Args:
            state: Current state (numpy array)
            action: Action to assess (int)
            q_values: Optional Q-values for all actions
            
        Returns:
            numpy array of features
        """
        features = []
        
        # 1. Agent stats - assuming last 3 elements of state are health, ammo, shields
        agent_health = state[-3]
        agent_ammo = state[-2]
        agent_shields = state[-1]
        
        features.extend([agent_health, agent_ammo, agent_shields])
        
        # 2. Action type (one-hot encoded)
        action_features = np.zeros(11)  # Assuming 11 action types
        if action < len(action_features):
            action_features[action] = 1
        features.extend(action_features)
        
        # 3. Action category features
        # Movement actions: 0-3, Combat actions: 4-7, Special actions: 8-10
        is_movement = 1 if action <= 3 else 0
        is_combat = 1 if 4 <= action <= 7 else 0
        is_special = 1 if action >= 8 else 0
        features.extend([is_movement, is_combat, is_special])
        
        # 4. Resource combination features
        # Low health flag
        low_health = 1 if agent_health < 0.3 else 0
        # Low ammo flag
        low_ammo = 1 if agent_ammo < 0.2 else 0
        # No shields flag
        no_shields = 1 if agent_shields <= 0 else 0
        
        features.extend([low_health, low_ammo, no_shields])
        
        # 5. Interaction features
        # Combat with low health
        combat_low_health = is_combat * low_health
        # Combat with low ammo
        combat_low_ammo = is_combat * low_ammo
        # Moving with critical health
        move_critical_health = is_movement * (1 if agent_health < 0.15 else 0)
        
        features.extend([combat_low_health, combat_low_ammo, move_critical_health])
        
        # 6. Q-value features if available
        if q_values is not None:
            # Current action Q-value
            action_q = q_values[action]
            # Best action and its Q-value
            best_action = np.argmax(q_values)
            best_q = q_values[best_action]
            # Difference from best Q-value
            q_diff = best_q - action_q
            # Normalized rank of this action (0 = best, 1 = worst)
            sorted_actions = np.argsort(-q_values)  # Descending sort
            rank = np.where(sorted_actions == action)[0][0]
            normalized_rank = rank / (len(q_values) - 1)
            
            features.extend([action_q, best_q, q_diff, normalized_rank])
            
            # Is this the best action?
            is_best_action = 1 if action == best_action else 0
            features.append(is_best_action)
        else:
            # Placeholders if Q-values not available
            features.extend([0, 0, 0, 0, 0])
        
        return np.array(features)
        
    def is_high_risk(self, state, action, q_values=None):
        """
        Determine if an action is high-risk
        
        Args:
            state: Current state
            action: Action to assess
            q_values: Optional Q-values for all actions
            
        Returns:
            (is_risky, reason): Tuple of boolean and explanation string
        """
        # Check feature cache first
        cache_key = (tuple(state.flatten()), action)
        current_time = time.time()
        
        if cache_key in self.feature_cache:
            entry_time, features = self.feature_cache[cache_key]
            if current_time - entry_time < self.cache_timeout:
                # Use cached features
                pass
            else:
                # Extract new features
                features = self.extract_features(state, action, q_values)
                self.feature_cache[cache_key] = (current_time, features)
        else:
            # Extract features
            features = self.extract_features(state, action, q_values)
            self.feature_cache[cache_key] = (current_time, features)
        
        # If we don't have a trained model, use a basic heuristic
        if not hasattr(self.model, 'predict_proba') or self.metrics["sample_count"] < 100:
            return self._fallback_risk_assessment(state, action, q_values, features)
            
        # Use model to predict risk
        features_reshaped = features.reshape(1, -1)
        
        try:
            # Get risk probability
            risk_prob = self.model.predict_proba(features_reshaped)[0][1]
            is_risky = risk_prob > 0.5
            
            # Generate explanation based on feature importances
            reason = self._generate_explanation(features, risk_prob)
            
            return is_risky, reason
        except Exception as e:
            # Fallback to basic assessment if model fails
            print(f"Model prediction failed: {str(e)}")
            return self._fallback_risk_assessment(state, action, q_values, features)
        
    def _fallback_risk_assessment(self, state, action, q_values, features):
        """Basic heuristic risk assessment when model is not ready"""
        # Extract agent stats
        agent_health = state[-3]
        agent_ammo = state[-2]
        agent_shields = state[-1]
        
        # Check for risky conditions
        is_risky = False
        reasons = []
        
        # Shooting with low ammo
        if action >= 4 and action <= 7 and agent_ammo < 0.2:
            is_risky = True
            reasons.append(f"Low ammo ({agent_ammo:.2f}) for shooting action")
            
        # Any action with very low health
        if agent_health < 0.15 and action != 9:  # 9 is "use cover"
            is_risky = True
            reasons.append(f"Critical health ({agent_health:.2f})")
            
        # Combat with low health
        if agent_health < 0.3 and action >= 4 and action <= 7:
            is_risky = True
            reasons.append(f"Low health ({agent_health:.2f}) for combat action")
            
        # Check Q-values if available
        if q_values is not None:
            action_q = q_values[action]
            best_action = np.argmax(q_values)
            best_q = q_values[best_action]
            
            # Action is significantly worse than best option
            if action != best_action and best_q > 0 and (best_q - action_q) > 0.5:
                is_risky = True
                reasons.append(f"Suboptimal action (Q={action_q:.2f} vs best={best_q:.2f})")
        
        reason = "; ".join(reasons) if reasons else "Low risk action"
        return is_risky, reason
        
    def _generate_explanation(self, features, risk_prob):
        """Generate human-readable explanation of risk assessment"""
        if not hasattr(self.model, 'named_steps'):
            return f"Risk probability: {risk_prob:.2f}"
            
        try:
            # Get feature importances from model if available
            classifier = self.model.named_steps['classifier']
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                # Map features to names
                feature_names = [
                    "Health", "Ammo", "Shields",
                    "Action0", "Action1", "Action2", "Action3", "Action4", "Action5",
                    "Action6", "Action7", "Action8", "Action9", "Action10",
                    "Is_Movement", "Is_Combat", "Is_Special",
                    "Low_Health", "Low_Ammo", "No_Shields",
                    "Combat_Low_Health", "Combat_Low_Ammo", "Move_Critical_Health",
                    "Action_Q", "Best_Q", "Q_Diff", "Normalized_Rank", "Is_Best_Action"
                ]
                
                # Keep only names that exist in our feature vector
                feature_names = feature_names[:len(features)]
                
                # Get top 3 most important features for this prediction
                top_indices = np.argsort(importances)[-3:]
                top_features = [(feature_names[i], importances[i], features[i]) for i in top_indices]
                
                # Generate explanation
                explanation_parts = []
                for name, importance, value in top_features:
                    explanation_parts.append(f"{name}={value:.2f} (importance={importance:.2f})")
                
                explanation = "Based on " + ", ".join(explanation_parts)
                return f"{explanation} → Risk probability: {risk_prob:.2f}"
            else:
                return f"Risk probability: {risk_prob:.2f}"
        except Exception as e:
            return f"Risk probability: {risk_prob:.2f}"
        
    def record_outcome(self, state, action, q_values, outcome, was_vetoed):
        """
        Record the outcome of an action to improve future risk assessment
        
        Args:
            state: State where action was taken
            action: Action that was taken
            q_values: Q-values at the time of decision
            outcome: Reward received after taking the action
            was_vetoed: Whether the action was vetoed
        """
        # Extract features
        features = self.extract_features(state, action, q_values)
        
        # Determine if this was actually risky (negative outcome)
        was_risky = outcome < 0
        
        # Store experience
        self.experience_buffer.append((features, was_risky))
        
        # Train model if we have enough data
        if len(self.experience_buffer) >= 100 and len(self.experience_buffer) % 50 == 0:
            self.train_model()
            
    def train_model(self):
        """Train the risk assessment model on collected experiences"""
        if len(self.experience_buffer) < 50:
            print("Not enough data to train risk model")
            return False
            
        # Convert buffer to training data
        X = np.array([features for features, _ in self.experience_buffer])
        y = np.array([label for _, label in self.experience_buffer])
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # (Re)create model if needed
        if self.model is None:
            self._create_model()
            
        # Train model
        try:
            self.model.fit(X_train, y_train)
            
            # Evaluate performance
            y_pred = self.model.predict(X_test)
            
            self.metrics["accuracy"] = accuracy_score(y_test, y_pred)
            self.metrics["precision"] = precision_score(y_test, y_pred, zero_division=0)
            self.metrics["recall"] = recall_score(y_test, y_pred, zero_division=0)
            self.metrics["f1_score"] = f1_score(y_test, y_pred, zero_division=0)
            self.metrics["sample_count"] = len(self.experience_buffer)
            
            print(f"Risk model trained on {len(self.experience_buffer)} samples:")
            print(f"  Accuracy: {self.metrics['accuracy']:.3f}")
            print(f"  Precision: {self.metrics['precision']:.3f}")
            print(f"  Recall: {self.metrics['recall']:.3f}")
            print(f"  F1 Score: {self.metrics['f1_score']:.3f}")
            
            return True
        except Exception as e:
            print(f"Error training risk model: {str(e)}")
            return False
            
    def save_model(self, path):
        """Save model to file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model, metrics, and a sample of experiences
        model_data = {
            'model': self.model,
            'metrics': self.metrics,
            'experiences': list(self.experience_buffer)[:100]  # Save 100 sample experiences
        }
        
        joblib.dump(model_data, path)
        print(f"Risk assessment model saved to {path}")
        
    def load_model(self, path):
        """Load model from file"""
        try:
            model_data = joblib.load(path)
            
            self.model = model_data['model']
            self.metrics = model_data.get('metrics', self.metrics)
            
            # Load sample experiences
            experiences = model_data.get('experiences', [])
            self.experience_buffer = deque(experiences, maxlen=10000)
            
            print(f"Risk assessment model loaded from {path}")
            print(f"  Accuracy: {self.metrics['accuracy']:.3f}")
            print(f"  Sample count: {self.metrics['sample_count']}")
            
            return True
        except Exception as e:
            print(f"Error loading risk model: {str(e)}")
            self._create_model()  # Create a new model
            return False
            
    def get_feature_importances(self):
        """
        Get feature importances from the model
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(self.model, 'named_steps'):
            return {}
            
        try:
            # Get feature importances from model if available
            classifier = self.model.named_steps['classifier']
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                # Map features to names
                feature_names = [
                    "Health", "Ammo", "Shields",
                    "Action0", "Action1", "Action2", "Action3", "Action4", "Action5",
                    "Action6", "Action7", "Action8", "Action9", "Action10",
                    "Is_Movement", "Is_Combat", "Is_Special",
                    "Low_Health", "Low_Ammo", "No_Shields",
                    "Combat_Low_Health", "Combat_Low_Ammo", "Move_Critical_Health",
                    "Action_Q", "Best_Q", "Q_Diff", "Normalized_Rank", "Is_Best_Action"
                ]
                
                # Keep only names that exist in our feature vector
                feature_names = feature_names[:len(importances)]
                
                # Create importance dictionary
                return {name: importance for name, importance in zip(feature_names, importances)}
                
            return {}
        except Exception as e:
            print(f"Error getting feature importances: {str(e)}")
            return {}