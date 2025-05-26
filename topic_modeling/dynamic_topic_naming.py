# dynamic_topic_naming.py
import logging
import numpy as np
import spacy
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model for word embeddings and POS tagging
try:
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    logger.warning(f"Error loading spaCy model: {e}. Falling back to simple heuristic for topic naming.")
    nlp = None

def simple_heuristic_naming(top_words, top_weights=None, num_words=2):
    """
    Generate a topic name using a simple heuristic: combine the top N words.
    
    Args:
        top_words (list): List of top words for the topic.
        top_weights (list, optional): List of weights corresponding to the top words.
        num_words (int): Number of top words to use in the name.
    
    Returns:
        str: Generated topic name.
    """
    try:
        if not top_words:
            return "Unknown Topic"
        # If weights are provided, sort words by weight
        if top_weights:
            word_weight_pairs = sorted(
                zip(top_words, top_weights),
                key=lambda x: x[1],
                reverse=True
            )
            top_words = [word for word, weight in word_weight_pairs]
        # Take the top N words (or fewer if not enough words)
        selected_words = top_words[:min(num_words, len(top_words))]
        # Capitalize each word and join with a space
        topic_name = " ".join(word.capitalize() for word in selected_words)
        return topic_name
    except Exception as e:
        logger.error(f"Error in simple_heuristic_naming: {e}")
        return "Unknown Topic"

def pos_based_naming(top_words, top_weights=None, num_words=5):
    """
    Generate a topic name using POS tagging to prioritize nouns and adjectives.
    
    Args:
        top_words (list): List of top words for the topic.
        top_weights (list, optional): List of weights corresponding to the top words.
        num_words (int): Number of top words to consider for naming.
    
    Returns:
        str: Generated topic name.
    """
    if not nlp:
        logger.warning("spaCy model not available. Falling back to simple heuristic.")
        return simple_heuristic_naming(top_words, top_weights, num_words=2)
    
    try:
        if not top_words:
            return "Unknown Topic"
        
        # Take the top N words (or fewer if not enough words)
        selected_words = top_words[:min(num_words, len(top_words))]
        if top_weights:
            selected_weights = top_weights[:min(num_words, len(top_weights))]
        else:
            selected_weights = [1.0 / len(selected_words)] * len(selected_words)
        
        # Normalize weights to sum to 1
        total_weight = sum(selected_weights)
        if total_weight > 0:
            selected_weights = [w / total_weight for w in selected_weights]
        else:
            selected_weights = [1.0 / len(selected_words)] * len(selected_words)
        
        # Use spaCy to get POS tags for the words
        word_pos_pairs = []
        for word, weight in zip(selected_words, selected_weights):
            doc = nlp(word)
            if doc:
                pos = doc[0].pos_  # Get the POS tag of the first token
                word_pos_pairs.append((word, pos, weight))
        
        # Prioritize nouns and adjectives
        nouns = [(word, weight) for word, pos, weight in word_pos_pairs if pos in ["NOUN", "PROPN"]]
        adjectives = [(word, weight) for word, pos, weight in word_pos_pairs if pos == "ADJ"]
        
        # Sort by weight
        nouns = sorted(nouns, key=lambda x: x[1], reverse=True)
        adjectives = sorted(adjectives, key=lambda x: x[1], reverse=True)
        
        # Construct the topic name: Adjective + Noun
        if adjectives and nouns:
            adj = adjectives[0][0].capitalize()
            noun = nouns[0][0].capitalize()
            return f"{adj} {noun}"
        elif nouns:
            # If no adjectives, use the top two nouns
            if len(nouns) >= 2:
                noun1 = nouns[0][0].capitalize()
                noun2 = nouns[1][0].capitalize()
                return f"{noun1} {noun2}"
            return nouns[0][0].capitalize()
        elif adjectives:
            # If no nouns, use the top adjective
            return adjectives[0][0].capitalize()
        else:
            # Fallback to simple heuristic if no nouns or adjectives
            logger.warning("No nouns or adjectives found. Falling back to simple heuristic.")
            return simple_heuristic_naming(top_words, top_weights, num_words=2)
    except Exception as e:
        logger.error(f"Error in pos_based_naming: {e}")
        return simple_heuristic_naming(top_words, top_weights, num_words=2)

def embedding_based_naming(top_words, top_weights=None, num_words=5):
    """
    Generate a topic name using word embeddings to find a central theme.
    
    Args:
        top_words (list): List of top words for the topic.
        top_weights (list, optional): List of weights corresponding to the top words.
        num_words (int): Number of top words to consider for embedding-based naming.
    
    Returns:
        str: Generated topic name.
    """
    if not nlp:
        logger.warning("spaCy model not available. Falling back to simple heuristic.")
        return simple_heuristic_naming(top_words, top_weights, num_words=2)
    
    try:
        if not top_words:
            return "Unknown Topic"
        
        # Take the top N words (or fewer if not enough words)
        selected_words = top_words[:min(num_words, len(top_words))]
        if top_weights:
            selected_weights = top_weights[:min(num_words, len(top_weights))]
        else:
            selected_weights = [1.0 / len(selected_words)] * len(selected_words)
        
        # Normalize weights to sum to 1
        total_weight = sum(selected_weights)
        if total_weight > 0:
            selected_weights = [w / total_weight for w in selected_weights]
        else:
            selected_weights = [1.0 / len(selected_words)] * len(selected_words)
        
        # Compute the weighted average of word embeddings
        weighted_vectors = []
        for word, weight in zip(selected_words, selected_weights):
            doc = nlp(word)
            if doc.vector_norm:  # Check if the word has a vector
                weighted_vectors.append(doc.vector * weight)
        
        if not weighted_vectors:
            logger.warning("No valid word vectors found. Falling back to simple heuristic.")
            return simple_heuristic_naming(top_words, top_weights, num_words=2)
        
        # Compute the centroid of the weighted vectors
        centroid = np.sum(weighted_vectors, axis=0) / len(weighted_vectors)
        
        # Find the word closest to the centroid
        similarities = []
        for word in selected_words:
            doc = nlp(word)
            if doc.vector_norm:
                similarity = doc.vector.dot(centroid) / (np.linalg.norm(doc.vector) * np.linalg.norm(centroid))
                similarities.append((word, similarity))
        
        if not similarities:
            logger.warning("No similarities computed. Falling back to simple heuristic.")
            return simple_heuristic_naming(top_words, top_weights, num_words=2)
        
        # Sort by similarity and take the most representative word
        similarities.sort(key=lambda x: x[1], reverse=True)
        representative_word = similarities[0][0]
        
        # Optionally, combine with the second most representative word
        if len(similarities) > 1:
            second_word = similarities[1][0]
            topic_name = f"{representative_word.capitalize()} {second_word.capitalize()}"
        else:
            topic_name = representative_word.capitalize()
        
        return topic_name
    except Exception as e:
        logger.error(f"Error in embedding_based_naming: {e}")
        return simple_heuristic_naming(top_words, top_weights, num_words=2)

def generate_topic_name(top_words, top_weights=None, method="pos", num_words=2):
    """
    Generate a topic name for a list of top words.
    
    Args:
        top_words (list): List of top words for the topic.
        top_weights (list, optional): List of weights corresponding to the top words.
        method (str): Method to use for naming ("simple", "embedding", or "pos").
        num_words (int): Number of words to consider for naming.
    
    Returns:
        str: Generated topic name.
    """
    if method == "embedding":
        return embedding_based_naming(top_words, top_weights, num_words)
    elif method == "pos":
        return pos_based_naming(top_words, top_weights, num_words)
    else:
        return simple_heuristic_naming(top_words, top_weights, num_words)