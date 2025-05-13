from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from recommendation_system import WellnessRecommendationSystem

app = Flask(__name__)

# Initialize the recommendation system
data_path = "Wellness Dataset - Dr.Fatma M. Talaat.csv"
recommender = WellnessRecommendationSystem(data_path)

# Initialize function
def initialize_model():
    if not os.path.exists('wellness_model.pkl'):
        print("Training new model...")
        recommender.load_data()
        recommender.preprocess_data()
        recommender.train_model()
        recommender.save_model()
    else:
        print("Loading existing model...")
        recommender.load_model()

# Call initialize function at startup
initialize_model()

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Get recommendation API endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get user input from form
        user_data = {
            'duration_of_sleep__hours_': float(request.form.get('sleep_hours', 7.0)),
            'level_of_physical_activity__minutes_per_day_': int(request.form.get('physical_activity', 30)),
            'level_of_stress__scale__1_10_': int(request.form.get('stress_level', 5)),
            'bmi_category': request.form.get('bmi_category', 'Normal'),
            'heart_rate__bpm_': int(request.form.get('heart_rate', 70)),
            'level_of_workload__scale__1_10_': int(request.form.get('workload', 5)),
            'weather': request.form.get('weather', 'sunny').lower()
        }
        
        # Get recommendation
        recommendation = recommender.get_recommendation(user_data)
        
        # Get specific recommendations based on type
        specific_recommendations = get_specific_recommendations(recommendation)
        
        return render_template(
            'recommendation.html',
            recommendation_type=recommendation,
            specific_recommendations=specific_recommendations
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# API endpoint for recommendations
@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    try:
        data = request.json
        recommendation = recommender.get_recommendation(data)
        specific_recommendations = get_specific_recommendations(recommendation)
        
        return jsonify({
            'recommendation_type': recommendation,
            'specific_recommendations': specific_recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_specific_recommendations(recommendation_type):
    """Get specific recommendations based on the recommendation type."""
    recommendations = {
        'meditation': [
            {
                'title': 'Mindfulness Meditation',
                'description': 'A 10-minute guided meditation focusing on breath awareness.',
                'link': 'https://www.mindful.org/how-to-meditate/'
            },
            {
                'title': 'Body Scan Meditation',
                'description': 'A relaxing meditation to release tension throughout your body.',
                'link': 'https://www.mindful.org/a-body-scan-meditation-to-help-you-sleep/'
            },
            {
                'title': 'Loving-Kindness Meditation',
                'description': 'Cultivate compassion and positive emotions with this practice.',
                'link': 'https://www.mindful.org/a-loving-kindness-meditation-to-boost-compassion/'
            }
        ],
        'workout': [
            {
                'title': 'Quick HIIT Workout',
                'description': '20-minute high-intensity interval training to boost energy.',
                'link': 'https://www.self.com/story/a-20-minute-hiit-workout-you-can-do-anywhere'
            },
            {
                'title': 'Yoga Flow',
                'description': 'A gentle 30-minute yoga sequence to improve flexibility and mood.',
                'link': 'https://www.yogajournal.com/practice/yoga-sequences/yoga-for-energy-morning-flow/'
            },
            {
                'title': 'Outdoor Walk/Run',
                'description': 'A 45-minute outdoor cardio session to clear your mind.',
                'link': 'https://www.runnersworld.com/training/a20807188/the-perfect-30-minute-workout/'
            }
        ],
        'product': [
            {
                'title': 'Stress Relief Tea',
                'description': 'Herbal tea blend with chamomile and lavender to reduce stress.',
                'link': 'https://www.healthline.com/nutrition/teas-that-help-you-sleep'
            },
            {
                'title': 'Essential Oil Diffuser',
                'description': 'Aromatherapy diffuser with calming essential oils for relaxation.',
                'link': 'https://www.healthline.com/health/essential-oils-for-anxiety'
            },
            {
                'title': 'Wellness Journal',
                'description': 'A guided journal to track your wellness journey and mood patterns.',
                'link': 'https://www.healthline.com/health/benefits-of-journaling'
            }
        ]
    }
    
    return recommendations.get(recommendation_type, recommendations['meditation'])

if __name__ == '__main__':
    app.run(debug=True)