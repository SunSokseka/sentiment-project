import spacy
nlp = spacy.load("en_core_web_sm")
# Lemmatization using spaCy
def lemmatize_tokens(tokens):
    """Lemmatizes tokens (reduces words to their root form) using spaCy."""
    return [token.lemma_ for token in nlp(" ".join(tokens))]