import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

def normalize_rating(rating):
    """Normalize rating from [1, 5] to [-1, 1]."""
    return (rating - 3) / 2

def get_sentiment_score(text):
    """Get sentiment score from review text using VADER."""
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']  # Returns value between -1 and 1

def combine_scores(rating_score, review_text_score, w_rating=0.4, w_text=0.6):
    """Combine normalized rating and review text sentiment scores."""
    return w_rating * rating_score + w_text * review_text_score

def assign_sentiment_label(combined_score):
    """Assign sentiment labels based on combined score."""
    if combined_score > 0.2:
        return 'Positive'
    elif combined_score < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

def annotate_data(df, rating_column="Rating", review_column="Review", w_rating=0.4, w_text=0.6):
    """Annotate dataset with sentiment labels."""
    
    df['normalized_rating'] = df[rating_column].apply(normalize_rating)
    df['review_text_score'] = df[review_column].apply(get_sentiment_score)
    df['combined_score'] = df.apply(lambda x: combine_scores(x['normalized_rating'], x['review_text_score'], w_rating, w_text), axis=1)
    df['sentiment_label'] = df['combined_score'].apply(assign_sentiment_label)
    
    return df['sentiment_label'].values
