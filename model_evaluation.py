import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def load_data(file_path='processed_wellness_data.csv'):
    """Load the processed wellness dataset."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def load_model(model_path='wellness_model.pkl'):
    """Load the trained wellness recommendation model."""
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def evaluate_model(model, df, test_size=0.2, random_state=42):
    """Evaluate the model and display detailed metrics."""
    # Extract features and target
    features = [
        'duration_of_sleep__hours_',
        'level_of_physical_activity__minutes_per_day_',
        'level_of_stress__scale__1_10_',
        'bmi_category',
        'heart_rate__bpm_',
        'level_of_workload__scale__1_10_',
        'weather'
    ]
    
    # Check if all features exist in the dataframe
    available_features = [col for col in features if col in df.columns]
    
    X = df[available_features]
    y = df['recommendation_type']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate precision for each class and weighted average
    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Calculate recall for each class and weighted average
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Calculate F1 score for each class and weighted average
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Get class labels
    classes = model.classes_ if hasattr(model, 'classes_') else np.unique(y)
    
    # Print metrics
    print("\n===== MODEL EVALUATION METRICS =====")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nPrecision by class:")
    for i, cls in enumerate(classes):
        print(f"  {cls}: {precision[i]:.4f}")
    print(f"Weighted Average Precision: {precision_weighted:.4f}")
    
    print("\nRecall by class:")
    for i, cls in enumerate(classes):
        print(f"  {cls}: {recall[i]:.4f}")
    print(f"Weighted Average Recall: {recall_weighted:.4f}")
    
    print("\nF1 Score by class:")
    for i, cls in enumerate(classes):
        print(f"  {cls}: {f1[i]:.4f}")
    print(f"Weighted Average F1 Score: {f1_weighted:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix_evaluation.png')
    print("Confusion matrix saved as 'confusion_matrix_evaluation.png'")
    
    # Plot precision, recall, and F1 score by class
    plt.figure(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1 Score')
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score by Class')
    plt.xticks(x, classes)
    plt.legend()
    plt.tight_layout()
    plt.savefig('class_metrics.png')
    print("Class metrics plot saved as 'class_metrics.png'")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'precision_weighted': precision_weighted,
        'recall': recall,
        'recall_weighted': recall_weighted,
        'f1': f1,
        'f1_weighted': f1_weighted,
        'classes': classes
    }

def feature_importance(model, df):
    """Visualize feature importance if available."""
    if not hasattr(model, 'named_steps') or not hasattr(model.named_steps['classifier'], 'feature_importances_'):
        print("Feature importance not available for this model")
        return
    
    # Get feature names
    feature_names = []
    preprocessor = model.named_steps['preprocessor']
    
    for name, trans, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            for i, col in enumerate(cols):
                categories = trans.categories_[i]
                feature_names.extend([f"{col}_{cat}" for cat in categories])
    
    # Get feature importances
    importances = model.named_steps['classifier'].feature_importances_
    
    # Ensure we have the right number of feature names
    if len(importances) != len(feature_names):
        print(f"Warning: Number of features ({len(importances)}) doesn't match feature names ({len(feature_names)})")
        feature_names = [f"Feature {i}" for i in range(len(importances))]
    
    # Sort features by importance
    indices = np.argsort(importances)[-20:]  # Top 20 features
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance_evaluation.png')
    print("Feature importance plot saved as 'feature_importance_evaluation.png'")

def main():
    """Main function to evaluate the model."""
    # Load data and model
    df = load_data()
    model = load_model()
    
    if df is not None and model is not None:
        # Evaluate model
        metrics = evaluate_model(model, df)
        
        # Visualize feature importance
        feature_importance(model, df)
        
        print("\nModel evaluation completed successfully!")
    else:
        print("Failed to load data or model. Please check file paths.")

if __name__ == "__main__":
    main()