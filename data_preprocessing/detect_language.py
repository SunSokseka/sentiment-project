# detect_language.py
from langdetect import detect_langs, DetectorFactory
import pandas as pd
import string
import re
try:
    import enchant
    ENCHANT_AVAILABLE = True
    enchant_dict = enchant.Dict("en_US")
except ImportError:
    ENCHANT_AVAILABLE = False
    enchant_dict = None

# Set seed for consistent language detection
DetectorFactory.seed = 0  

# Expanded list of common English words, including sentiment-related words
COMMON_ENGLISH_WORDS = {
    'i', 'you', 'the', 'a', 'to', 'and', 'is', 'in', 'it', 'of', 'for', 'with', 'on', 'at',
    'this', 'but', 'from', 'by', 'like', 'your', 'my', 'they', 'we', 'are', 'was', 'were',
    'good', 'bad', 'great', 'awesome', 'terrible', 'ok', 'okay', 'nice', 'poor', 'excellent',
    'horrible', 'wonderful', 'amazing', 'worst', 'best', 'love', 'hate', 'dislike', 'enjoy',
    'happy', 'sad', 'angry', 'fine', 'perfect', 'awful', 'not', 'very', 'so', 'too'
}

def normalize_text(text):
    """Normalize text by fixing common contractions and standardizing spaces."""
    if not text or not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Fix common contractions
    contractions = {
        "dont": "don't",
        "cant": "can't",
        "wont": "won't",
        "isnt": "isn't",
        "arent": "aren't",
        "didnt": "didn't",
        "havent": "haven't",
        "hasnt": "hasn't",
        "wouldnt": "wouldn't",
        "shouldnt": "shouldn't",
        "couldnt": "couldn't"
    }
    for contraction, expanded in contractions.items():
        text = re.sub(r'\b' + contraction + r'\b', expanded, text)
    # Remove punctuation and extra spaces
    text = ''.join(char for char in text if char not in string.punctuation).strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def detect_english(text):
    """
    Detects if a given text is in English with improved reliability for short text.
    
    Args:
        text (str): The text to analyze.
    
    Returns:
        bool: True if the text is detected as English, False otherwise.
    """
    # Handle empty or invalid text
    if not text or pd.isna(text) or text.lower() in ["none", "extent"]:
        return False
    
    # Normalize the text
    text = normalize_text(text)
    if not text:
        return False
    
    # Check ASCII ratio as a secondary indicator
    ascii_ratio = sum(ord(char) < 128 for char in text) / len(text)
    words = text.split()
    word_count = len(words)
    
    # Special handling for very short text (1-2 words)
    if word_count <= 2:
        # Check if the word(s) are in the common English words list
        common_words_ratio = len(set(words).intersection(COMMON_ENGLISH_WORDS)) / word_count if word_count else 0
        if common_words_ratio >= 0.5:  # Be lenient for short text
            return True
        
        # Check if the word exists in an English dictionary (if enchant is available)
        if ENCHANT_AVAILABLE and word_count == 1:
            if enchant_dict.check(words[0]):
                return True
        
        # For short text, be more lenient with ASCII check
        if ascii_ratio > 0.7:  # Lowered threshold for short text
            return True
        return False
    
    # For longer text, try langdetect
    try:
        langs = detect_langs(text)
        print(f"Debug: Language probabilities for '{text}': {langs}")  # Debug output
        for lang in langs:
            if lang.lang == 'en' and lang.prob > 0.3:  # Lowered threshold
                return True
    except:
        pass  # Proceed to fallback checks
    
    # Fallback 1: If mostly ASCII, assume English
    if ascii_ratio > 0.8:
        return True
    
    # Fallback 2: Check for common English words
    common_words_ratio = len(set(words).intersection(COMMON_ENGLISH_WORDS)) / word_count if word_count else 0
    if common_words_ratio > 0.5:
        return True
    
    return False

def filter_english_reviews(df, text_column="Review"):
    """
    Filters the dataset to keep only English-language reviews.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing reviews.
        text_column (str): Column name of the reviews.
    
    Returns:
        pd.DataFrame: DataFrame with only English reviews.
    """
    df["is_english"] = df[text_column].astype(str).apply(detect_english)
    return df[df["is_english"]].drop(columns=["is_english"])

# Example usage:
# df = pd.read_csv("reviews.csv")
# df_filtered = filter_english_reviews(df)
# df_filtered.to_csv("cleaned_reviews.csv", index=False)