                                                                    🧘‍♀️ Wellness Recommendation System
                                                                        ✍️ Author: Sweta Sharma
📌 Overview
The Wellness Recommendation System is a machine learning-based application that provides personalized wellness recommendations by analyzing user health data. It evaluates various health metrics—such as sleep duration, physical activity, stress levels, BMI, heart rate, workload, and environmental conditions—to suggest one of three wellness strategies:

🧘‍♂️ Meditation

🏋️ Workout

🛍️ Product Suggestion

🌟 Key Features
✅ Personalized Recommendations based on real-time health inputs

🖥️ Flask Web Interface for user-friendly interaction

🔗 REST API Endpoint for external integration

🌲 Random Forest Classifier for robust predictions

📊 Data Visualization for insights and model transparency

🗂️ Project Structure
File/Folder	Description
app.py	Flask server for the web interface and API
recommendation_system.py	Core engine for training and prediction
data_preprocessing.py	Utilities for data cleaning and preparation
train_model.py	Script for model training with hyperparameter tuning
model_evaluation.py	Model performance metrics and visualizations
processed_wellness_data.csv	Cleaned and ready-to-use dataset
wellness_model.pkl	Trained machine learning model
requirements.txt	List of required Python packages

🚀 Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/wellness-recommendation-system.git
cd wellness-recommendation-system
Install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
🧪 Usage
▶️ Run the Web Application
bash
Copy
Edit
python app.py
Visit http://localhost:5000 in your browser.

🔗 Get Recommendations via API
POST to /api/recommend with JSON:

json
Copy
Edit
{
  "duration_of_sleep__hours_": 7.5,
  "level_of_physical_activity__minutes_per_day_": 30,
  "level_of_stress__scale__1_10_": 5,
  "bmi_category": "Normal",
  "heart_rate__bpm_": 72,
  "level_of_workload__scale__1_10_": 6,
  "weather": "sunny"
}
🧠 Train a New Model
bash
Copy
Edit
python train_model.py
📈 Evaluate Model Performance
bash
Copy
Edit
python model_evaluation.py
🧬 Model Details
Algorithm Used: Random Forest Classifier
Input Features:

Sleep duration (hours)

Physical activity (minutes/day)

Stress level (scale 1–10)

BMI category: Underweight, Normal, Overweight, Obese

Heart rate (BPM)

Workload (scale 1–10)

Weather condition (e.g., sunny, rainy)

🎯 Recommendation Types
Type	When It's Recommended
🧘 Meditation	High stress or poor sleep quality
🏋️ Workout	Low physical activity or weight concerns
🛍️ Product	Needs related to hydration or nutrition

🧹 Data Preprocessing Includes:
Column name cleaning

Weather feature extraction

Blood pressure parsing

Mood-to-recommendation mapping

Handling missing data

Standardizing numerical values

One-hot encoding of categorical data

📊 Model Evaluation Metrics
Accuracy Score

Precision, Recall, F1-Score (per class)

Confusion Matrix

Feature Importance Visualization

📦 Dependencies
nginx
Copy
Edit
pandas >= 1.5.3  
numpy >= 1.26.0  
scikit-learn >= 1.2.2  
joblib >= 1.2.0  
matplotlib >= 3.7.1  
seaborn >= 0.12.2  
streamlit >= 1.22.0  
nltk >= 3.8.1  
textblob >= 0.17.1  
Install them using:

bash
Copy
Edit
pip install -r requirements.txt
📬 Contact
For suggestions or queries, feel free to reach out to Sweta Sharma via GitHub
