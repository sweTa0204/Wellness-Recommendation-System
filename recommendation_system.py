import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import re

class WellnessRecommendationSystem:
    def __init__(self, data_path):
        """Initialize the recommendation system with the dataset path."""
        self.data_path = data_path
        self.data = None
        self.model = None
        self.preprocessor = None
        self.features = None
        self.target = None
        
    def load_data(self):
        """Load the dataset from the specified path."""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully with {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
            
            # Clean column names (remove spaces and special characters)
            self.data.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col.lower().strip()) for col in self.data.columns]
            
            # Extract weather from environmental aspects
            self.data['weather'] = self.data['environmental_aspects__such_as_weather_and_air_quality_'].apply(
                lambda x: x.split(',')[0].strip().lower() if isinstance(x, str) else 'unknown'
            )
            
            # Map mood to recommendation type
            self.data['recommendation_type'] = self.data['mood_output'].apply(self.map_mood_to_recommendation)
            
            print("\nSample data:")
            print(self.data.head())
            print("\nData info:")
            print(self.data.info())
            
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def map_mood_to_recommendation(self, mood):
        """Map mood to recommendation type."""
        if mood.lower() == 'happy':
            return 'workout'
        elif mood.lower() == 'sad':
            return 'meditation'
        else:  # neutral or other moods
            return 'product'
    
    def preprocess_data(self):
        """Preprocess the data for model training."""
        if self.data is None:
            print("Please load the data first.")
            return False
        
        try:
            # Select relevant features
            selected_features = [
                'duration_of_sleep__hours_',
                'level_of_physical_activity__minutes_per_day_',
                'level_of_stress__scale__1_10_',
                'bmi_category',
                'heart_rate__bpm_',
                'level_of_workload__scale__1_10_',
                'weather'
            ]
            
            # Check if all selected features exist in the dataframe
            available_features = [col for col in selected_features if col in self.data.columns]
            
            # Identify numerical and categorical features
            numerical_features = self.data[available_features].select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = self.data[available_features].select_dtypes(include=['object']).columns.tolist()
            
            # Create preprocessor
            numerical_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            
            # Prepare features and target
            self.features = self.data[available_features]
            self.target = self.data['recommendation_type']
            
            print("Data preprocessing completed.")
            return True
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return False
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train the recommendation model."""
        if self.features is None or self.target is None:
            print("Please preprocess the data first.")
            return False
        
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                self.features, self.target, test_size=test_size, random_state=random_state
            )
            
            # Create and train the model
            self.model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=random_state))
            ])
            
            self.model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model trained with accuracy: {accuracy:.2f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def save_model(self, model_path='wellness_model.pkl'):
        """Save the trained model to disk."""
        if self.model is None:
            print("Please train the model first.")
            return False
        
        try:
            joblib.dump(self.model, model_path)
            print(f"Model saved to {model_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path='wellness_model.pkl'):
        """Load a trained model from disk."""
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_recommendation(self, user_data):
        """Get wellness recommendations based on user data."""
        if self.model is None:
            print("Please train or load a model first.")
            return None
        
        try:
            # Convert user_data to DataFrame if it's a dictionary
            if isinstance(user_data, dict):
                user_data = pd.DataFrame([user_data])
            
            # Extract relevant features
            features_to_use = [col for col in self.features.columns if col in user_data.columns]
            
            # If we don't have all the features, fill in with defaults
            for col in self.features.columns:
                if col not in user_data.columns:
                    if col in ['duration_of_sleep__hours_']:
                        user_data[col] = 7.0  # Default sleep hours
                    elif col in ['level_of_physical_activity__minutes_per_day_']:
                        user_data[col] = 30  # Default physical activity
                    elif col in ['level_of_stress__scale__1_10_']:
                        user_data[col] = user_data.get('stress_level', 5)  # Use stress_level if available
                    elif col in ['heart_rate__bpm_']:
                        user_data[col] = user_data.get('heart_rate', 70)  # Use heart_rate if available
                    elif col in ['level_of_workload__scale__1_10_']:
                        user_data[col] = 5  # Default workload
                    elif col in ['bmi_category']:
                        user_data[col] = 'Normal'  # Default BMI category
                    elif col in ['weather']:
                        user_data[col] = user_data.get('weather', 'sunny')  # Use weather if available
            
            # Make prediction
            recommendation = self.model.predict(user_data[self.features.columns])
            return recommendation[0]
        except Exception as e:
            print(f"Error getting recommendation: {e}")
            return "meditation"  # Default recommendation if error occurs

# Example usage
if __name__ == "__main__":
    # Adjust the path to your dataset
    data_path = "Wellness Dataset - Dr.Fatma M. Talaat.csv"
    
    # Initialize and train the recommendation system
    recommender = WellnessRecommendationSystem(data_path)
    
    if recommender.load_data():
        if recommender.preprocess_data():
            if recommender.train_model():
                recommender.save_model()
                
                # Example of getting a recommendation
                user_data = {
                    'mood': 'stressed',
                    'weather': 'rainy',
                    'heart_rate': 75,
                    'sleep_hours': 6,
                    'stress_level': 7,
                    'duration_of_sleep__hours_': 6,
                    'level_of_physical_activity__minutes_per_day_': 30,
                    'level_of_stress__scale__1_10_': 7,
                    'bmi_category': 'Normal',
                    'heart_rate__bpm_': 75,
                    'level_of_workload__scale__1_10_': 6
                }
                
                recommendation = recommender.get_recommendation(user_data)
                print(f"Recommended wellness activity: {recommendation}")