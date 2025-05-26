import os
import sys
# Add the root directory of the project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..")))

from sentiment_analysis.data_preprocessing.text_cleaning import clean_text
from sentiment_analysis.data_preprocessing.tokenizer import tokenize_text
# from sentiment_analysis.data_preprocessing.text_cleaning import lemmatizer
from sentiment_analysis.data_preprocessing.remove_stopword import remove_stopwords
from sentiment_analysis.data_preprocessing.lemmatize import lemmatize_tokens

# Apply Preprocessing
def preprocess_text(text):
    """Full text preprocessing: cleaning, tokenization, stopwords removal, and lemmatization."""
    cleaned_text = clean_text(text)  # Step 1: Clean text
    tokens = tokenize_text(cleaned_text)  # Step 2: Tokenization
    filtered_tokens = remove_stopwords(tokens)  # Step 3: Remove stopwords
    lemmatized_tokens = lemmatize_tokens(filtered_tokens)  # Step 4: Lemmatization

    # Ensure lemmatized_tokens is returned as a string, not a list
    if isinstance(lemmatized_tokens, list):
        return " ".join(lemmatized_tokens)
    
    return lemmatized_tokens  # In case it already returns a string
