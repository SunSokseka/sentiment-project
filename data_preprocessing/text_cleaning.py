import re
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Text Cleaning
def clean_text(text):
    """Cleans text by removing special characters, punctuation, and converting to lowercase."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text



