import re
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Keep words that are important
custom_stopwords = STOP_WORDS - {"not", "no", "never", "very", "well", "super", "almost", "unless"}

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Stopword removal using spaCy
def remove_stopwords(tokens):
    """Removes stopwords from tokenized text using custom stopwords."""
    return [token.text for token in nlp(" ".join(tokens)) if token.text.lower() not in custom_stopwords]