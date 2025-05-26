# lda.py
import logging
import numpy as np
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from topic_modeling.dynamic_topic_naming import generate_topic_name

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_text(text, clean_text, tokenize_text, remove_stopwords, lemmatize_tokens):
    """
    Preprocess text for LDA: cleaning, tokenization, stopword removal, and lemmatization.
    
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

def train_lda_model(texts, num_topics=5, passes=10):
    """
    Train an LDA model on the given texts.
    
    Args:
        texts (list of str): List of preprocessed texts.
        num_topics (int): Number of topics to identify.
        passes (int): Number of passes through the corpus during training.
    
    Returns:
        tuple: (LdaModel, Dictionary, list) - Trained LDA model, dictionary, and corpus.
    """
    try:
        # Tokenize the texts
        tokenized_texts = [text.split() for text in texts if text]
        if not tokenized_texts:
            raise ValueError("No valid texts provided for LDA training.")
        # Create a dictionary and corpus
        dictionary = Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        # Train the LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            random_state=42
        )
        return lda_model, dictionary, corpus
    except Exception as e:
        logger.error(f"Error training LDA model: {e}")
        return None, None, None

def get_topic_lda(text, lda_model, dictionary):
    """
    Assign a topic to a single text using the trained LDA model.
    
    Args:
        text (str): Preprocessed text to assign a topic to.
        lda_model (LdaModel): Trained LDA model.
        dictionary (Dictionary): Gensim dictionary used for the LDA model.
    
    Returns:
        int: Topic ID with the highest probability, or None if assignment fails.
    """
    if not text or not isinstance(text, str):
        return None
    try:
        bow = dictionary.doc2bow(text.split())
        topic_probs = lda_model.get_document_topics(bow)
        return max(topic_probs, key=lambda x: x[1])[0] if topic_probs else None
    except Exception as e:
        logger.error(f"Error in get_topic_lda for text '{text}': {e}")
        return None

def compute_lda_coherence(lda_model, texts, dictionary):
    """
    Compute the coherence score for the LDA model.
    
    Args:
        lda_model (LdaModel): Trained LDA model.
        texts (list of str): List of preprocessed texts.
        dictionary (Dictionary): Gensim dictionary used for the LDA model.
    
    Returns:
        float: Coherence score (C_V metric), or None if computation fails.
    """
    try:
        tokenized_texts = [text.split() for text in texts if text]
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        return coherence_score
    except Exception as e:
        logger.error(f"Error computing LDA coherence: {e}")
        return None

def get_top_words_lda(lda_model, num_words=10):
    """
    Get the top words and their weights for each topic in the LDA model.
    
    Args:
        lda_model (LdaModel): Trained LDA model.
        num_words (int): Number of top words to return per topic.
    
    Returns:
        dict: Dictionary mapping topic IDs to tuples of (top words, weights).
    """
    try:
        topics = {}
        for topic_id in range(lda_model.num_topics):
            terms = lda_model.show_topic(topic_id, topn=num_words)
            words = [term[0] for term in terms]
            weights = [term[1] for term in terms]
            topics[topic_id] = (words, weights)
        return topics
    except Exception as e:
        logger.error(f"Error getting top words for LDA: {e}")
        return {}

def generate_lda_topic_names(lda_model, num_words=10, naming_method="simple", num_name_words=2):
    """
    Generate topic names for LDA topics using dynamic naming.
    
    Args:
        lda_model (LdaModel): Trained LDA model.
        num_words (int): Number of top words to consider for naming.
        naming_method (str): Method for naming ("simple" or "embedding").
        num_name_words (int): Number of words to use in the topic name (for simple method).
    
    Returns:
        dict: Dictionary mapping topic IDs to generated topic names.
    """
    try:
        topic_names = {}
        topics = get_top_words_lda(lda_model, num_words=num_words)
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
        logger.error(f"Error generating LDA topic names: {e}")
        return {}