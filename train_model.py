import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess_data import prepare_features_for_modeling

def train_mood_prediction_model(df, features, target, preprocessor):
    """Train a model to predict mood based on wellness features."""
    # Split the data
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define hyperparameters for grid search
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Evaluate the model
    y_pred = best_model.predict(X_test)
    
    print("Best parameters:", grid_search.best_params_)
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Feature importance
    if hasattr(best_model['classifier'], 'feature_importances_'):
        # Get feature names after preprocessing
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                for col in cols:
                    feature_names.extend([f"{col}_{cat}" for cat in trans.categories_[0]])
        
        # Get feature importances
        importances = best_model['classifier'].feature_importances_
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        indices = np.argsort(importances)[-20:]  # Top 20 features
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    # Save the model
    joblib.dump(best_model, 'mood_prediction_model.pkl')
    
    return best_model

def train_recommendation_model(df):
    """Train a model to recommend wellness activities based on user data."""
    # Create recommendation categories
    df['recommendation_category'] = 'Unknown'
    
    # Meditation recommendations for high stress, poor sleep, or anxiety
    df.loc[(df['level_of_stress_scale:_1–10'] > 6) | 
           (df['duration_of_sleep_hours'] < 6.5), 'recommendation_category'] = 'Meditation'
    
    # Workout recommendations for low physical activity or weight management needs
    df.loc[(df['level_of_physical_activity_minutes_per_day'] < 30) | 
           (df['bmi_category'].isin(['Overweight', 'Obese'])), 'recommendation_category'] = 'Workout'
    
    # Product recommendations for specific health needs
    df.loc[(df['water_intake_liters'] < 2.0) | 
           (df['steps_numeric'] < 5000) | 
           (df['diet_type'].isin(['unhealthy', 'high in fat', 'high in carbs'])), 'recommendation_category'] = 'Product'
    
    # Prepare features for the recommendation model
    features, _, preprocessor = prepare_features_for_modeling(df)
    
    # Split the data
    X = df[features]
    y = df['recommendation_category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    
    print("\nRecommendation Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(pipeline, 'recommendation_model.pkl')
    
    return pipeline

if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv("preprocessed_wellness_data.csv")
    
    # Prepare features for modeling
    features, target, preprocessor = prepare_features_for_modeling(df)
    
    # Train mood prediction model
    mood_model = train_mood_prediction_model(df, features, target, preprocessor)
    
    # Train recommendation model
    recommendation_model = train_recommendation_model(df)
    
    print("Model training completed successfully!")