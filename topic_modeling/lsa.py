# lsa.py
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from topic_modeling.dynamic_topic_naming import generate_topic_name

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_text(text, clean_text, tokenize_text, remove_stopwords, lemmatize_tokens):
    """
    Preprocess text for LSA: cleaning, tokenization, stopword removal, and lemmatization.
    
    Args:
        text (str): Input text to preprocess.
        clean_text (callable): Function to clean text.
        tokenize_text (callable): Function to tokenize text.
        remove_stopwords (callable): Function to remove stopwords.
        lemmatize_tokens (callable): Function to lemmatize tokens.
    
    Returns:
        str: Preprocessed text as a space-separated string of tokens, or None if preprocessing fails.
    """
    if not text or not isinstance(text, str):
        return None
    try:
        cleaned_text = clean_text(text)
        tokens = tokenize_text(cleaned_text)
        filtered_tokens = remove_stopwords(tokens)
        lemmatized_tokens = lemmatize_tokens(filtered_tokens)
        if isinstance(lemmatized_tokens, list):
            return " ".join(lemmatized_tokens)
        return lemmatized_tokens
    except Exception as e:
        logger.error(f"Error in preprocess_text for text '{text}': {e}")
        return None

def train_lsa_model(texts, num_topics=5):
    """
    Train an LSA model on the given texts using TF-IDF and SVD.
    
    Args:
        texts (list of str): List of preprocessed texts.
        num_topics (int): Number of topics to identify.
    
    Returns:
        tuple: (TruncatedSVD, TfidfVectorizer, numpy.ndarray) - Trained LSA model, vectorizer, and topic matrix.
    """
    try:
        # Create a document-term matrix using TF-IDF
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        X = vectorizer.fit_transform(texts)
        # Apply SVD
        lsa_model = TruncatedSVD(n_components=num_topics, random_state=42)
        lsa_topic_matrix = lsa_model.fit_transform(X)
        return lsa_model, vectorizer, lsa_topic_matrix
    except Exception as e:
        logger.error(f"Error training LSA model: {e}")
        return None, None, None

def get_topic_lsa(text, lsa_model, vectorizer):
    """
    Assign a topic to a single text using the trained LSA model.
    
    Args:
        text (str): Preprocessed text to assign a topic to.
        lsa_model (TruncatedSVD): Trained LSA model.
        vectorizer (TfidfVectorizer): TF-IDF vectorizer used for the LSA model.
    
    Returns:
        int: Topic ID with the highest weight, or None if assignment fails.
    """
    if not text or not isinstance(text, str):
        return None
    try:
        tfidf_features = vectorizer.transform([text])
        topic_distribution = lsa_model.transform(tfidf_features)
        topic = np.argmax(topic_distribution, axis=1)[0]
        return topic
    except Exception as e:
        logger.error(f"Error in get_topic_lsa for text '{text}': {e}")
        return None

def compute_lsa_reconstruction_error(texts, vectorizer, lsa_model):
    """
    Compute the reconstruction error for the LSA model.
    
    Args:
        texts (list of str): List of preprocessed texts.
        vectorizer (TfidfVectorizer): TF-IDF vectorizer used for the LSA model.
        lsa_model (TruncatedSVD): Trained LSA model.
    
    Returns:
        float: Reconstruction error (mean squared error), or None if computation fails.
    """
    try:
        tfidf_matrix = vectorizer.transform(texts)
        W = lsa_model.transform(tfidf_matrix)
        H = lsa_model.components_
        reconstructed = np.dot(W, H)
        error = mean_squared_error(tfidf_matrix.toarray(), reconstructed)
        return error
    except Exception as e:
        logger.error(f"Error computing LSA reconstruction error: {e}")
        return None

def get_top_words_lsa(lsa_model, vectorizer, num_words=10):
    """
    Get the top words and their weights for each topic in the LSA model.
    
    Args:
        lsa_model (TruncatedSVD): Trained LSA model.
        vectorizer (TfidfVectorizer): TF-IDF vectorizer used for the LSA model.
        num_words (int): Number of top words to return per topic.
    
    Returns:
        dict: Dictionary mapping topic IDs to tuples of (top words, weights).
    """
    try:
        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        for topic_id in range(lsa_model.n_components):
            top_term_indices = lsa_model.components_[topic_id].argsort()[-num_words:][::-1]
            words = [feature_names[i] for i in top_term_indices]
            weights = [lsa_model.components_[topic_id][i] for i in top_term_indices]
            topics[topic_id] = (words, weights)
        return topics
    except Exception as e:
        logger.error(f"Error getting top words for LSA: {e}")
        return {}

def generate_lsa_topic_names(lsa_model, vectorizer, num_words=10, naming_method="simple", num_name_words=2):
    """
    Generate topic names for LSA topics using dynamic naming.
    
    Args:
        lsa_model (TruncatedSVD): Trained LSA model.
        vectorizer (TfidfVectorizer): TF-IDF vectorizer used for the LSA model.
        num_words (int): Number of top words to consider for naming.
        naming_method (str): Method for naming ("simple" or "embedding").
        num_name_words (int): Number of words to use in the topic name (for simple method).
    
    Returns:
        dict: Dictionary mapping topic IDs to generated topic names.
    """
    try:
        topic_names = {}
        topics = get_top_words_lsa(lsa_model, vectorizer, num_words=num_words)
        for topic_id, (words, weights) in topics.items():
            topic_name = generate_topic_name(
                top_words=words,
                top_weights=weights,
                method=naming_method,
                num_words=num_name_words
            )
            topic_names[topic_id] = topic_name
        return topic_names
    except Exception as e:
        logger.error(f"Error generating LSA topic names: {e}")
        return {}