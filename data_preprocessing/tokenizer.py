import spacy

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Tokenization using spaCy
def tokenize_text(text):
    """Tokenizes text into words using spaCy."""
    doc = nlp(text)  # Process text with spaCy
    return [token.text for token in doc if not token.is_punct]  # Exclude punctuation
