# e:\sentiment_analysis\topic_modeling\NMF.py
import os  # Added missing import
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import joblib

class NMFTopicModel:
    def __init__(self, num_topics=10, max_features=5000):
        """
        Initialize the NMF topic model with specified parameters.
        """
        self.num_topics = num_topics
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=self.max_features)
        self.nmf_model = NMF(n_components=self.num_topics, random_state=42)
        self.W = None  # Document-topic matrix
        self.H = None  # Topic-word matrix
        self.feature_names = None
        self.topic_names = None

    def preprocess_data(self, documents):
        """
        Convert text data to a TF-IDF matrix.
        """
        X = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return X

    def fit(self, documents):
        """
        Fit the NMF model on the provided text data.
        """
        X = self.preprocess_data(documents)
        self.W = self.nmf_model.fit_transform(X)
        self.H = self.nmf_model.components_

    def get_top_words(self, num_words=10):
        """
        Extract the top words for each topic.
        """
        topic_keywords = {}
        for topic_idx, topic in enumerate(self.H):
            top_words = [self.feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
            topic_keywords[topic_idx] = top_words
        return topic_keywords

    def assign_topic_names(self, topic_names):
        """
        Assign meaningful names to topics.
        """
        self.topic_names = topic_names

    def predict_topics(self, documents):
        """
        Predict the dominant topic for each document.
        """
        X = self.vectorizer.transform(documents)
        W_new = self.nmf_model.transform(X)
        topic_assignments = W_new.argmax(axis=1)

        df_results = pd.DataFrame({'Feedback': documents, 'Topic': topic_assignments})
        if self.topic_names:
            df_results['Topic Name'] = df_results['Topic'].map(self.topic_names)
        return df_results

    def save_model(self, filepath):
        """
        Save the trained model to a file using joblib.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'vectorizer': self.vectorizer,
            'nmf_model': self.nmf_model,
            'W': self.W,
            'H': self.H,
            'feature_names': self.feature_names,
            'topic_names': self.topic_names
        }, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """
        Load the trained model from a file using joblib.
        """
        data = joblib.load(filepath)
        model = cls(num_topics=data['nmf_model'].n_components, max_features=data['vectorizer'].max_features)
        model.vectorizer = data['vectorizer']
        model.nmf_model = data['nmf_model']
        model.W = data['W']
        model.H = data['H']
        model.feature_names = data['feature_names']
        model.topic_names = data['topic_names']
        return model

if __name__ == "__main__":
    print("NMF Topic Model Module Loaded.")