import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download necessary NLTK resources (first-time only)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def clean_column_names(df):
    """Clean column names by removing spaces and special characters."""
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col.lower().strip()) for col in df.columns]
    return df

def extract_weather(df, column_name='environmental_aspects__such_as_weather_and_air_quality_'):
    """Extract weather information from environmental aspects column."""
    if column_name in df.columns:
        df['weather'] = df[column_name].apply(
            lambda x: x.split(',')[0].strip().lower() if isinstance(x, str) else 'unknown'
        )
    return df

def extract_blood_pressure(df, column_name='systolic_and_diastolic_blood_pressure'):
    """Extract systolic and diastolic blood pressure as separate columns."""
    if column_name in df.columns:
        df['systolic_bp'] = df[column_name].apply(
            lambda x: int(x.split('/')[0]) if isinstance(x, str) and '/' in x else np.nan
        )
        df['diastolic_bp'] = df[column_name].apply(
            lambda x: int(x.split('/')[1]) if isinstance(x, str) and '/' in x else np.nan
        )
    return df

def map_mood_to_recommendation(mood):
    """Map mood to recommendation type."""
    if isinstance(mood, str):
        if mood.lower() == 'happy':
            return 'workout'
        elif mood.lower() == 'sad':
            return 'meditation'
        else:  # neutral or other moods
            return 'product'
    return 'meditation'  # Default

# New sentiment analysis functions
def analyze_sentiment_textblob(text):
    """Analyze sentiment of text using TextBlob.
    
    Returns:
        dict: Dictionary containing polarity (-1 to 1) and subjectivity (0 to 1)
    """
    if not isinstance(text, str) or not text.strip():
        return {'polarity': 0, 'subjectivity': 0}
    
    analysis = TextBlob(text)
    return {
        'polarity': analysis.sentiment.polarity,
        'subjectivity': analysis.sentiment.subjectivity
    }

def analyze_sentiment_vader(text):
    """Analyze sentiment of text using VADER.
    
    Returns:
        dict: Dictionary containing compound score and individual scores
    """
    if not isinstance(text, str) or not text.strip():
        return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
    
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)

def add_sentiment_features(df, text_column):
    """Add sentiment analysis features to the dataframe.
    
    Args:
        df: DataFrame containing the data
        text_column: Column name containing text to analyze
    
    Returns:
        DataFrame with added sentiment features
    """
    if text_column not in df.columns:
        print(f"Column '{text_column}' not found in dataframe")
        return df
    
    # Apply TextBlob sentiment analysis
    sentiment_data = df[text_column].apply(analyze_sentiment_textblob)
    df['sentiment_polarity'] = sentiment_data.apply(lambda x: x['polarity'])
    df['sentiment_subjectivity'] = sentiment_data.apply(lambda x: x['subjectivity'])
    
    # Apply VADER sentiment analysis
    vader_data = df[text_column].apply(analyze_sentiment_vader)
    df['sentiment_compound'] = vader_data.apply(lambda x: x['compound'])
    df['sentiment_positive'] = vader_data.apply(lambda x: x['pos'])
    df['sentiment_neutral'] = vader_data.apply(lambda x: x['neu'])
    df['sentiment_negative'] = vader_data.apply(lambda x: x['neg'])
    
    # Add sentiment category
    df['sentiment_category'] = df['sentiment_compound'].apply(
        lambda score: 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral')
    )
    
    return df

def preprocess_dataset(file_path):
    """Preprocess the wellness dataset."""
    try:
        # Load the data
        df = pd.read_csv(file_path)
        print(f"Dataset loaded with shape: {df.shape}")
        
        # Clean column names
        df = clean_column_names(df)
        
        # Extract weather information
        df = extract_weather(df)
        
        # Extract blood pressure
        df = extract_blood_pressure(df)
        
        # Add sentiment analysis if text columns exist
        text_columns = ['comments', 'feedback', 'notes']
        for col in text_columns:
            if col in df.columns:
                df = add_sentiment_features(df, col)
                print(f"Added sentiment features based on '{col}' column")
        
        # Map mood to recommendation type
        if 'mood_output' in df.columns:
            df['recommendation_type'] = df['mood_output'].apply(map_mood_to_recommendation)
        
        # Handle missing values
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col] = df[col].fillna(df[col].median())
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        print("Preprocessing completed successfully!")
        return df
    
    except Exception as e:
        print(f"Error preprocessing dataset: {e}")
        return None

# Function to analyze sentiment of arbitrary text
def analyze_text_sentiment(text):
    """Analyze sentiment of arbitrary text input.
    
    Args:
        text: String text to analyze
    
    Returns:
        dict: Dictionary containing sentiment analysis results
    """
    if not isinstance(text, str) or not text.strip():
        return {
            'textblob': {'polarity': 0, 'subjectivity': 0},
            'vader': {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0},
            'category': 'neutral'
        }
    
    # TextBlob analysis
    textblob_result = analyze_sentiment_textblob(text)
    
    # VADER analysis
    vader_result = analyze_sentiment_vader(text)
    
    # Determine category based on VADER compound score
    category = 'positive' if vader_result['compound'] >= 0.05 else (
        'negative' if vader_result['compound'] <= -0.05 else 'neutral'
    )
    
    return {
        'textblob': textblob_result,
        'vader': vader_result,
        'category': category
    }

def visualize_data(df, output_file='data_visualization.png'):
    """Create visualizations for the wellness dataset."""
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Distribution of moods
    plt.subplot(2, 2, 1)
    if 'mood_output' in df.columns:
        sns.countplot(x='mood_output', data=df)
        plt.title('Distribution of Moods')
        plt.xticks(rotation=45)
    
    # Plot 2: Distribution of recommendation types
    plt.subplot(2, 2, 2)
    if 'recommendation_type' in df.columns:
        sns.countplot(x='recommendation_type', data=df)
        plt.title('Distribution of Recommendation Types')
        plt.xticks(rotation=45)
    
    # Plot 3: Sleep duration vs. Stress level
    plt.subplot(2, 2, 3)
    if 'duration_of_sleep__hours_' in df.columns and 'level_of_stress__scale__1_10_' in df.columns:
        sns.scatterplot(x='duration_of_sleep__hours_', y='level_of_stress__scale__1_10_', 
                        hue='mood_output' if 'mood_output' in df.columns else None, data=df)
        plt.title('Sleep Duration vs. Stress Level')
        plt.xlabel('Sleep Duration (hours)')
        plt.ylabel('Stress Level (1-10)')
    
    # Plot 4: Physical activity vs. Heart rate
    plt.subplot(2, 2, 4)
    if 'level_of_physical_activity__minutes_per_day_' in df.columns and 'heart_rate__bpm_' in df.columns:
        sns.scatterplot(x='level_of_physical_activity__minutes_per_day_', y='heart_rate__bpm_', 
                        hue='mood_output' if 'mood_output' in df.columns else None, data=df)
        plt.title('Physical Activity vs. Heart Rate')
        plt.xlabel('Physical Activity (minutes/day)')
        plt.ylabel('Heart Rate (BPM)')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Visualizations saved to {output_file}")

# New function to visualize sentiment analysis results
def visualize_sentiment(df, output_file='sentiment_visualization.png'):
    """Create visualizations for sentiment analysis results."""
    if 'sentiment_category' not in df.columns:
        print("No sentiment analysis data found in dataframe")
        return
    
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Distribution of sentiment categories
    plt.subplot(2, 2, 1)
    sns.countplot(x='sentiment_category', data=df)
    plt.title('Distribution of Sentiment Categories')
    
    # Plot 2: Sentiment polarity distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df['sentiment_polarity'], kde=True)
    plt.title('Distribution of Sentiment Polarity')
    
    # Plot 3: Sentiment vs. Recommendation Type
    plt.subplot(2, 2, 3)
    if 'recommendation_type' in df.columns:
        cross_tab = pd.crosstab(df['sentiment_category'], df['recommendation_type'])
        cross_tab.plot(kind='bar', stacked=True)
        plt.title('Sentiment Category vs. Recommendation Type')
        plt.xlabel('Sentiment Category')
        plt.ylabel('Count')
    
    # Plot 4: Sentiment Compound Score vs. Stress Level
    plt.subplot(2, 2, 4)
    if 'level_of_stress__scale__1_10_' in df.columns:
        sns.scatterplot(x='sentiment_compound', y='level_of_stress__scale__1_10_', data=df)
        plt.title('Sentiment Compound Score vs. Stress Level')
        plt.xlabel('Sentiment Compound Score')
        plt.ylabel('Stress Level (1-10)')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Sentiment visualizations saved to {output_file}")

if __name__ == "__main__":
    # Preprocess the dataset
    file_path = "Wellness Dataset - Dr.Fatma M. Talaat.csv"
    processed_df = preprocess_dataset(file_path)
    
    if processed_df is not None:
        # Visualize the data
        visualize_data(processed_df)
        
        # Visualize sentiment analysis if available
        if any(col.startswith('sentiment_') for col in processed_df.columns):
            visualize_sentiment(processed_df)
        
        # Save the processed dataset
        processed_df.to_csv("processed_wellness_data.csv", index=False)
        print("Processed dataset saved to 'processed_wellness_data.csv'")