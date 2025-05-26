import os
import sys
import pandas as pd
import spacy
# Add the root directory of the project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))
from sentiment_analysis.data_preprocessing.word_embedding import vectorize_text
from sentiment_analysis.data_preprocessing.processing_text import preprocess_text


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load Data
def load_data(filepath):
    """Load the CSV file into a DataFrame."""
    return pd.read_csv(filepath)

# Preprocess Entire Dataset
def preprocess_data(filepath):
    """Loads and preprocesses the dataset."""
    df = load_data(filepath)
    
    # Ensure the dataset has a column named "Review"
    if "Review" not in df.columns:
        raise ValueError("Dataset must have a 'Review' column")

    # Apply full text preprocessing pipeline
    df["cleaned_review"] = df["Review"].astype(str).apply(preprocess_text)

    # Convert to Word Embeddings
    df["embedding"] = df["cleaned_review"].apply(lambda x: vectorize_text(x))  
    return df

# Save Preprocessed Data
def save_preprocessed_data(df, output_filepath):
    """Saves preprocessed data to CSV."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)  # Ensure directory exists
    df.to_csv(output_filepath, index=False, encoding="utf-8-sig")  # Ensure correct encoding
    print(f"Preprocessed data saved to: {output_filepath}")
