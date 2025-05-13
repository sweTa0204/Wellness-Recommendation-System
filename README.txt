# Wellness Recommendation System
## Author:
Sweta Sharma
## Overview
The Wellness Recommendation System is a machine learning-based application that provides personalized wellness recommendations based on user health data. The system analyzes various health metrics such as sleep duration, physical activity, stress levels, BMI, heart rate, workload, and environmental factors to suggest one of three recommendation types: meditation, workout, or product.

## Features
- **Personalized Recommendations**: Get tailored wellness suggestions based on your health metrics
- **Web Application Interface**: Easy-to-use Flask web interface for inputting health data
- **API Endpoint**: REST API for integration with other applications
- **Machine Learning Model**: Random Forest classifier trained on wellness data
- **Data Visualization**: Visual analysis of wellness data and model performance

## Project Structure
- `app.py`: Flask web application for serving recommendations
- `recommendation_system.py`: Core recommendation engine with model training and prediction
- `data_preprocessing.py`: Data cleaning and preparation utilities
- `train_model.py`: Model training scripts with hyperparameter tuning
- `model_evaluation.py`: Comprehensive model evaluation metrics and visualizations
- `processed_wellness_data.csv`: Preprocessed dataset
- `wellness_model.pkl`: Trained machine learning model
- `requirements.txt`: Project dependencies

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application
```bash
python app.py
```
This will start the Flask server at http://localhost:5000

### Getting Recommendations via API
Send a POST request to `/api/recommend` with the following JSON structure:
```json
{
  "duration_of_sleep__hours_": 7.5,
  "level_of_physical_activity__minutes_per_day_": 30,
  "level_of_stress__scale__1_10_": 5,
  "bmi_category": "Normal",
  "heart_rate__bpm_": 72,
  "level_of_workload__scale__1_10_": 6,
  "weather": "sunny"
}
```

### Training a New Model
```bash
python train_model.py
```

### Evaluating Model Performance
```bash
python model_evaluation.py
```

## Model Details
The recommendation system uses a Random Forest classifier with the following features:
- Sleep duration (hours)
- Physical activity level (minutes per day)
- Stress level (scale 1-10)
- BMI category (Underweight, Normal, Overweight, Obese)
- Heart rate (BPM)
- Workload level (scale 1-10)
- Weather conditions

## Recommendation Types
1. **Meditation**: Recommended for users with high stress levels or poor sleep quality
2. **Workout**: Suggested for users with low physical activity or weight management needs
3. **Product**: Recommended for specific health needs like hydration or nutrition

## Data Preprocessing
The system preprocesses raw wellness data by:
- Cleaning column names
- Extracting weather information
- Parsing blood pressure readings
- Mapping mood states to recommendation types
- Handling missing values
- Standardizing numerical features
- One-hot encoding categorical features

## Model Evaluation
The model is evaluated using:
- Accuracy score
- Precision, recall, and F1 score for each class
- Confusion matrix
- Feature importance analysis

## Dependencies
- pandas >= 1.5.3
- numpy >= 1.26.0
- scikit-learn >= 1.2.2
- joblib >= 1.2.0
- matplotlib >= 3.7.1
- seaborn >= 0.12.2
- streamlit >= 1.22.0
- nltk>=3.8.1
- textblob>=0.17.1



        