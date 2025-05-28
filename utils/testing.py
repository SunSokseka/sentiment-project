import os
import sys
import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import transformers  # Add this import to access transformers.__version__
from transformers import BertForSequenceClassification, BertTokenizer
import gensim
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
import pickle
from datetime import datetime, timedelta
import dateparser
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from functools import lru_cache
import matplotlib
import pkg_resources
from langdetect import detect_langs, DetectorFactory
import spacy
import string
import re
import tensorflow  # Already added for tensorflow.__version__
import requests  # Already added for S3 URL loading
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
# Try to import pyenchant for dictionary lookup
try:
    import enchant
    ENCHANT_AVAILABLE = True
    enchant_dict = enchant.Dict("en_US")
except ImportError:
    ENCHANT_AVAILABLE = False
    enchant_dict = None

# Set page config as the first Streamlit command
st.set_page_config(layout="wide")

# Add both current and parent directory to sys.path for better compatibility
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

# Import Preprocessing Functions
from data_preprocessing.text_cleaning import clean_text
from data_preprocessing.tokenizer import tokenize_text
from data_preprocessing.remove_stopword import remove_stopwords
from data_preprocessing.lemmatize import lemmatize_tokens
from data_preprocessing.detect_language import detect_english

# List to collect error messages during model loading
error_messages = []

# Load spacy model once at the start
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # Disable unused components for speed

# Ensure consistent results with langdetect
DetectorFactory.seed = 0

# Expanded list of common English words, including sentiment-related words
COMMON_ENGLISH_WORDS = {
    'i', 'you', 'the', 'a', 'to', 'and', 'is', 'in', 'it', 'of', 'for', 'with', 'on', 'at',
    'this', 'but', 'from', 'by', 'like', 'your', 'my', 'they', 'we', 'are', 'was', 'were',
    'good', 'bad', 'great', 'awesome', 'terrible', 'ok', 'okay', 'nice', 'poor', 'excellent',
    'horrible', 'wonderful', 'amazing', 'worst', 'best', 'love', 'hate', 'dislike', 'enjoy',
    'happy', 'sad', 'angry', 'fine', 'perfect', 'awful', 'not', 'very', 'so', 'too'
}

# Function to check if a date string is already in a standard format
def is_standard_date_format(date_str):
    """Check if the date string is already in a pandas-recognizable format."""
    if not date_str or not isinstance(date_str, str):
        return False
    try:
        pd.to_datetime(date_str, errors='raise')
        return True
    except (ValueError, TypeError):
        return False

# Cache the results of convert_relative_date
@lru_cache(maxsize=1000)
def convert_relative_date(date_str):
    """Convert relative date strings like 'about a year ago' to standard YYYY-MM-DD format."""
    if not date_str or not isinstance(date_str, str):
        return None
    parsed_date = dateparser.parse(date_str)
    return parsed_date.strftime('%Y-%m-%d') if parsed_date else None

# Batch date conversion
def batch_convert_dates(date_series):
    to_convert = date_series.apply(lambda x: not is_standard_date_format(str(x)))
    result = date_series.copy()
    if to_convert.any():
        non_standard_dates = date_series[to_convert].apply(str)
        converted_dates = non_standard_dates.apply(convert_relative_date)
        result[to_convert] = converted_dates
    return result

# Text normalization for language detection
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

# Batch language detection
def batch_detect_english(texts):
    """Detect English language for a list of texts."""
    def is_english(text):
        if not text or pd.isna(text) or text.lower() in ["none", "extent"]:
            return False
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
    return np.array([is_english(text) for text in texts])

# Try to import transformers components with error handling
try:
    from transformers import BertForSequenceClassification, BertTokenizer
    print("Transformers imported successfully.")
except ImportError as e:
    error_messages.append(f"Failed to import transformers components: {e}. Please update transformers with 'pip install --upgrade transformers' and ensure torch is installed.")
    st.stop()

# Load BERT Sentiment Model
bert_model_path = "./sentiment_model"
bert_model = None
bert_tokenizer = None
try:
    print("bert_model_path: ", bert_model_path)
    bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
    print("import")
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    bert_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    print("BERT Model loaded successfully.")
except Exception as e:
    error_messages.append(f"Error loading BERT model: {e}. Check the model path and ensure compatibility with the transformers version.")
    st.stop()

# Load BiLSTM Model and Tokenizer with detailed error handling
bilstm_model_path = "./sentiment_model/bilstm.h5"
tokenizer_path = "./sentiment_model/tokenizer.pkl"
bilstm_model = None
bilstm_tokenizer = None
try:
    # Check if files exist
    if not os.path.exists(bilstm_model_path):
        raise FileNotFoundError(f"BiLSTM model file not found at {bilstm_model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")

    # Check file sizes (to detect if they're empty)
    if os.path.getsize(bilstm_model_path) == 0:
        raise ValueError(f"BiLSTM model file at {bilstm_model_path} is empty")
    if os.path.getsize(tokenizer_path) == 0:
        raise ValueError(f"Tokenizer file at {tokenizer_path} is empty")

    # Load the BiLSTM model
    bilstm_model = load_model(bilstm_model_path)
    if bilstm_model is None:
        raise ValueError("Failed to load BiLSTM model: load_model returned None")
    print("BiLSTM model loaded successfully.")

    # Load the tokenizer with additional validation
    with open(tokenizer_path, 'rb') as handle:
        bilstm_tokenizer = pickle.load(handle)
    
    # Validate that the tokenizer is not None
    if bilstm_tokenizer is None:
        raise ValueError("Failed to load tokenizer: pickle.load returned None")
    
    # Validate the tokenizer's attributes
    if not hasattr(bilstm_tokenizer, 'word_index'):
        raise ValueError("Loaded tokenizer is invalid: missing word_index attribute")
    if bilstm_tokenizer.word_index is None:
        raise ValueError("Loaded tokenizer is invalid: word_index is None")
    if not bilstm_tokenizer.word_index:
        raise ValueError("Loaded tokenizer is invalid: word_index is empty")
    
    print("BiLSTM Model and Tokenizer loaded successfully.")
except FileNotFoundError as e:
    error_messages.append(f"BiLSTM model or tokenizer not loaded: {e}. The BiLSTM option will be unavailable.")
except ValueError as e:
    error_messages.append(f"BiLSTM model or tokenizer not loaded: {e}. The BiLSTM option will be unavailable.")
except Exception as e:
    error_messages.append(f"Unexpected error loading BiLSTM model or tokenizer: {e}. The BiLSTM option will be unavailable.")
    # For debugging: Print the stack trace to the console
    import traceback
    traceback.print_exc()

# Load LDA Model, Dictionary, and Topic Names
lda_model_path = r"./models/lda_model"
dictionary_path = r"./models/lda_dictionary"
topic_names_path = r"./models/lda_topic_names.txt"

lda_model = None
dictionary = None
topic_names = {}

try:
    if not os.path.exists(lda_model_path):
        raise FileNotFoundError(f"LDA model file not found at {lda_model_path}")
    if not os.path.exists(dictionary_path):
        raise FileNotFoundError(f"Dictionary file not found at {dictionary_path}")
    if not os.path.exists(topic_names_path):
        raise FileNotFoundError(f"Topic names file not found at {topic_names_path}")

    # Load LDA model and dictionary
    lda_model = LdaModel.load(lda_model_path)
    dictionary = Dictionary.load(dictionary_path)

    # Load topic names
    with open(topic_names_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    parts = line.strip().split(': ', 1)
                    if len(parts) != 2:
                        error_messages.append(f"Skipping malformed topic name line: {line.strip()}")
                        continue
                    topic_label = parts[0].strip()
                    if not topic_label.startswith("Topic "):
                        error_messages.append(f"Skipping malformed topic label: {topic_label}")
                        continue
                    topic_id_str = topic_label.replace("Topic ", "")
                    topic_id = int(topic_id_str)
                    topic_name = parts[1].strip()
                    topic_names[topic_id] = topic_name
                except ValueError as e:
                    error_messages.append(f"Error parsing topic ID from line '{line.strip()}': {e}")
                    continue
    print("LDA Model, Dictionary, and Topic Names loaded successfully.")
    print("Topic Names:", topic_names)
except FileNotFoundError as e:
    error_messages.append(f"Failed to load LDA components: {e}. Topic modeling will be unavailable.")
    lda_model = None
    dictionary = None
    topic_names = {}
except Exception as e:
    error_messages.append(f"Unexpected error loading LDA components: {e}. Topic modeling will be unavailable.")
    lda_model = None
    dictionary = None
    topic_names = {}

# Sentiment Mapping for Ratings (for English reviews)
def map_rating_to_score(rating):
    """Map a rating (1-5) to a numerical sentiment score (-1 to 1) using linear interpolation."""
    if pd.isna(rating):
        return 0.0
    try:
        rating = float(rating)
        # Ensure rating is within the valid range (1 to 5)
        rating = max(1, min(5, rating))
        # Linearly interpolate: rating 1 -> -1, rating 3 -> 0, rating 5 -> 1
        score = (rating - 3) / 2
        return score
    except (ValueError, TypeError):
        return 0.0

# Direct Rating to Sentiment Mapping (for non-English reviews)
def rating_to_sentiment(rating):
    """Map a rating (1-5) directly to a sentiment label for non-English reviews."""
    if pd.isna(rating):
        return "neutral"
    try:
        rating = float(rating)
        if rating <= 2.5:
            return "negative"
        elif rating <= 3.5:
            return "neutral"
        else:
            return "positive"
    except (ValueError, TypeError):
        return "neutral"

# Batch BERT Sentiment Prediction
def batch_predict_sentiment_bert(texts, batch_size=16):
    """Predict sentiment for a list of texts using BERT in batches."""
    if not texts:
        return [[0.0, 0.0, 0.0]] * len(texts)
    
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_texts = [text if text and not pd.isna(text) else "" for text in batch_texts]
        inputs = bert_tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probabilities.tolist())
    return all_probs

# Preprocess Text for BiLSTM
def preprocess_text(text):
    """Preprocess a single text for BiLSTM input."""
    if not text or pd.isna(text) or text.lower() in ["none", "extent"]:
        return ""
    # Step 1: Clean the text
    cleaned_text = clean_text(text)
    # Step 2: Process with spacy
    doc = nlp(cleaned_text)
    # Step 3: Tokenize, remove stopwords, and lemmatize
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    # Step 4: Join tokens into a single string
    return " ".join(tokens)

# Predict Sentiment with BiLSTM (Return Probabilities)
def predict_sentiment_bilstm(text, max_length=100):
    if bilstm_model is None or bilstm_tokenizer is None or not text or pd.isna(text):
        return [0.0, 0.0, 0.0]
    processed_text = preprocess_text(text)
    if not processed_text:  # Handle empty processed text
        return [0.0, 0.0, 0.0]
    sequences = bilstm_tokenizer.texts_to_sequences([processed_text])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    prediction = bilstm_model.predict(padded_sequences, verbose=0)
    probabilities = prediction[0].tolist()
    return probabilities

# Calculate Model Sentiment Score from Probabilities
def calculate_model_score(probabilities):
    """Convert model probabilities to a sentiment score (-1 to 1)."""
    sentiment_values = [-1, 0, 1]
    score = sum(p * v for p, v in zip(probabilities, sentiment_values))
    return score

# Combine Model and Rating Scores (for English reviews)
def combine_scores(model_score, rating_score, model_weight=0.7, rating_weight=0.3):
    """Combine model and rating scores into a final score."""
    if model_score is None or rating_score is None:
        return 0.0
    combined_score = (model_weight * model_score + rating_weight * rating_score) / (model_weight + rating_weight)
    return combined_score

# Map Combined Score to Sentiment Label (for English reviews)
def score_to_sentiment(score):
    """Map a numerical score (-1 to 1) to a sentiment label."""
    if score < -0.33:
        return "negative"
    elif score > 0.33:
        return "positive"
    else:
        return "neutral"

# Batch Text Preprocessing
def batch_preprocess_text(texts):
    """Preprocess a list of texts in batch."""
    texts = [text if text and not pd.isna(text) and text.lower() not in ["none", "extent"] else "" for text in texts]
    cleaned_texts = [clean_text(text) for text in texts]
    docs = list(nlp.pipe(cleaned_texts))
    processed_tokens = []
    for doc in docs:
        tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
        lemmatized_tokens = [token.lemma_ for token in doc if token.text in tokens]
        processed_tokens.append(lemmatized_tokens)
    return processed_tokens

# Batch Topic Assignment
def batch_get_topic(processed_tokens_list):
    """Assign topics to a list of preprocessed tokens in batch."""
    if not lda_model or not dictionary:
        return [None] * len(processed_tokens_list)
    
    topics = []
    for tokens in processed_tokens_list:
        if not tokens:
            topics.append(None)
            continue
        try:
            bow = dictionary.doc2bow(tokens)
            topic_probs = lda_model.get_document_topics(bow)
            if topic_probs:
                topics.append(max(topic_probs, key=lambda x: x[1])[0])
            else:
                topics.append(None)
        except Exception as e:
            error_messages.append(f"Error assigning topic: {e}")
            topics.append(None)
    return topics

# Generate Bubble Chart for a Given Topic (with adjective filter option)
def generate_bubble_chart(topic_id, adjective_only=False):
    if not lda_model:
        st.warning("LDA model not available for bubble chart.")
        return None
    
    # Get topic terms from the LDA model
    topic_terms = lda_model.show_topic(topic_id, topn=20)
    terms = [term[0] for term in topic_terms]
    weights = [term[1] for term in topic_terms]
    
    # Filter for adjectives if adjective_only is True
    if adjective_only:
        # Process terms with spaCy to identify adjectives
        doc = nlp(" ".join(terms))
        adjective_terms = [token.text for token in doc if token.pos_ == "ADJ"]
        if not adjective_terms:
            st.warning("No adjectives found in the topic terms.")
            return None
        # Filter original terms and weights to include only adjectives
        filtered_terms = []
        filtered_weights = []
        for term, weight in zip(terms, weights):
            if term in adjective_terms:
                filtered_terms.append(term)
                filtered_weights.append(weight)
        terms = filtered_terms
        weights = filtered_weights
    
    # Define bubble sizes based on word length
    min_size = 700
    max_size = 2000
    word_lengths = [len(term) for term in terms]
    if not word_lengths:  # Handle case where no terms remain after filtering
        st.warning("No terms available for bubble chart after filtering.")
        return None
    max_length = max(word_lengths)
    min_length = min(word_lengths)
    bubble_sizes = [
        min_size + (len(term) - min_length) / (max_length - min_length) * (max_size - min_size)
        for term in terms
    ]
    
    # Generate random positions with collision avoidance
    np.random.seed(42)
    x = np.random.uniform(-7, 7, len(terms))
    y = np.random.uniform(-7, 7, len(terms))
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 2.0:
                factor = (2.0 - dist) / 2.0
                x[i] += factor * (dx / dist) * 0.5
                x[j] -= factor * (dx / dist) * 0.5
                y[i] += factor * (dy / dist) * 0.5
                y[j] -= factor * (dy / dist) * 0.5
    
    # Create the bubble chart
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')
    ax.set_facecolor('black')
    for i, (term, size, x_pos, y_pos) in enumerate(zip(terms, bubble_sizes, x, y)):
        ax.scatter(x_pos, y_pos, s=size, alpha=0.6, c='limegreen', edgecolors='white', linewidth=0.5)
        max_font_size = 12
        min_font_size = 6
        font_size = max(min_font_size, max_font_size * (max_length - len(term)) / (max_length - min_length + 1))
        ax.text(x_pos, y_pos, term, ha='center', va='center', fontsize=font_size, color='white', weight='bold')
        percentage = f"{weights[i] * 100:.2f}%"
        ax.text(x_pos, y_pos - 0.5, percentage, ha='center', va='center', fontsize=8, color='white')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-9, 9)
    ax.set_ylim(-9, 9)
    title = f"Topic {topic_id}: {topic_names.get(topic_id, 'Unknown')}"
    if adjective_only:
        title += " (Adjectives Only)"
    ax.set_title(title, color='white', fontsize=12, pad=10)
    return fig

# Function to apply background colors to the Sentiment column
def highlight_sentiment(val):
    if val == 'positive':
        return 'background-color: green; color: white'
    elif val == 'negative':
        return 'background-color: red; color: white'
    elif val == 'neutral':
        return 'background-color: yellow; color: black'
    return ''

# Display any error messages collected during model loading
if error_messages:
    for msg in error_messages:
        st.error(msg)
# Function to save figure as PNG and return the file path
def save_figure(fig, filename):
    fig.savefig(filename, bbox_inches='tight', format='png')
    plt.close(fig)
    return filename

# Function to generate PDF with images
def generate_pdf(image_files, output_filename="visualizations_report.pdf"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y_position = height - 50  # Start position from top

    for image_file in image_files:
        if os.path.exists(image_file):
            img = ImageReader(image_file)
            img_width, img_height = img.getSize()
            aspect = img_height / float(img_width)
            # Scale image to fit page width
            img_display_width = width - 100
            img_display_height = img_display_width * aspect
            if y_position - img_display_height < 50:  # Check if there's space on the page
                c.showPage()
                y_position = height - 50
            c.drawImage(img, 50, y_position - img_display_height, img_display_width, img_display_height)
            y_position -= (img_display_height + 20)  # Add spacing

    c.save()
    buffer.seek(0)
    return buffer, output_filename
# Streamlit UI
st.title("📊 Sentiment Analysis Dashboard")

# Sidebar
st.sidebar.header("🔍 Filters")
model_options = ["BERT"]
if bilstm_model is not None and bilstm_tokenizer is not None:
    model_options.append("BiLSTM")
model_option = st.sidebar.radio("Select Model:", model_options)
option = st.sidebar.radio("Choose input type:", ("Text Input", "Upload File"))

if option == "Text Input":
    user_input = st.text_area("Enter a review:")
    rating_input = st.slider("Enter a rating (1-5):", min_value=1, max_value=5, value=3)
    if st.button("Analyze"):
        if user_input.strip():
            if detect_english(user_input):
                if model_option == "BERT":
                    probabilities = batch_predict_sentiment_bert([user_input])[0]
                elif model_option == "BiLSTM":
                    probabilities = predict_sentiment_bilstm(user_input)
                model_score = calculate_model_score(probabilities)
                rating_score = map_rating_to_score(rating_input)
                combined_score = combine_scores(model_score, rating_score)
                final_sentiment = score_to_sentiment(combined_score)
                processed_tokens = batch_preprocess_text([user_input])[0]
                topic = batch_get_topic([processed_tokens])[0]
                topic_name = topic_names.get(topic, "Unknown")
                st.success(f"**Predicted Sentiment (Combined):** {final_sentiment}")
                st.info(f"**Model Score:** {model_score:.2f}")
                st.info(f"**Rating Score:** {rating_score:.2f}")
                st.info(f"**Combined Score:** {combined_score:.2f}")
                st.info(f"**Identified Topic:** {topic_name}")
            else:
                rating_score = map_rating_to_score(rating_input)
                final_sentiment = rating_to_sentiment(rating_input)  # Use direct mapping for non-English
                processed_tokens = batch_preprocess_text([user_input])[0]
                topic = batch_get_topic([processed_tokens])[0]
                topic_name = topic_names.get(topic, "Unknown")
                st.success(f"**Predicted Sentiment (Based on Rating):** {final_sentiment}")
                st.info(f"**Rating Score:** {rating_score:.2f}")
                st.info(f"**Identified Topic:** {topic_name}")
                st.warning("⚠ This text is not in English. Sentiment is based on the rating provided.")
        else:
            st.warning("⚠ Please enter some text.")

if option == "Upload File":
    st.markdown("#### Expected CSV Format")
    st.markdown("The CSV should have columns for reviews, company, date, and rating. You will map these columns below after uploading. Download a sample CSV below:")
    sample_data = pd.DataFrame({
        "Review": ["Great product!", "Not good", "Average experience"],
        "Company": ["CompanyA", "CompanyB", "CompanyA"],
        "Date": ["2023-01-01", "2023-02-01", "2023-03-01"],
        "Rating": [5, 2, 3]
    })
    sample_csv = sample_data.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Sample CSV", sample_csv, "sample_data.csv", "text/csv")
    
    uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Column Mapping
        st.sidebar.subheader("📝 Column Mapping")
        st.sidebar.markdown("Map your CSV columns to the required fields:")
        csv_columns = ["-- Select a column --"] + list(df.columns)
        
        text_col = st.sidebar.selectbox("Select Review Column", csv_columns, key="text_col")
        company_col = st.sidebar.selectbox("Select Company Column", csv_columns, key="company_col")
        date_col = st.sidebar.selectbox("Select Date Column", csv_columns, key="date_col")
        rating_col = st.sidebar.selectbox("Select Rating Column", csv_columns, key="rating_col")

        # Validate column mapping
        selected_columns = [text_col, company_col, date_col, rating_col]
        if "-- Select a column --" in selected_columns:
            st.error("⚠ Please map all required columns.")
            st.stop()
        
        # Check for duplicate selections
        selected_columns_set = set(selected_columns)
        if len(selected_columns_set) != len(selected_columns):
            st.error("⚠ Each column must be mapped to a unique CSV column. Please check your selections.")
            st.stop()

        with st.spinner("Processing your data... This may take a moment."):
            # Preprocess the input data using mapped columns
            df[text_col] = df[text_col].apply(lambda x: "" if str(x).lower() in ["none", "extent"] else x)
            df[company_col] = df[company_col].fillna("Unknown")
            df[date_col] = batch_convert_dates(df[date_col])
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            median_date = df[date_col].median()
            df[date_col].fillna(median_date, inplace=True)
            df[date_col] = pd.to_datetime(df[date_col]).dt.date  # Strip time component
            
            df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
            df = df.dropna(subset=[rating_col])
            df[rating_col] = df[rating_col].astype(float)  # Keep as float to handle decimals like 2.4
            
            # Detect language for all reviews
            df["Is_English"] = batch_detect_english(df[text_col].tolist())
            
            # Calculate rating score for all reviews (for English reviews and display purposes)
            df["Rating_Score"] = df[rating_col].apply(map_rating_to_score)
            
            # Predict sentiment probabilities for English reviews only
            if model_option == "BERT":
                english_texts = df[df["Is_English"]][text_col].tolist()
                if english_texts:
                    english_probs = batch_predict_sentiment_bert(english_texts)
                else:
                    english_probs = []
                # Map probabilities back to the DataFrame
                prob_dict = {i: prob for i, prob in zip(df[df["Is_English"]].index, english_probs)}
                df["Model_Probabilities"] = df.index.map(lambda i: prob_dict.get(i, [0.0, 0.0, 0.0]))
            elif model_option == "BiLSTM":
                df["Model_Probabilities"] = df.apply(
                    lambda row: predict_sentiment_bilstm(row[text_col]) if row["Is_English"] else [0.0, 0.0, 0.0], axis=1
                )
            
            # Calculate model score for English reviews
            df["Model_Score"] = df["Model_Probabilities"].apply(calculate_model_score)
            
            # Combine scores for English reviews; for non-English reviews, Rating_Score is just for display
            df["Combined_Score"] = df.apply(
                lambda row: combine_scores(row["Model_Score"], row["Rating_Score"]) if row["Is_English"] else row["Rating_Score"],
                axis=1
            )
            
            # Map to sentiment: use score_to_sentiment for English, rating_to_sentiment for non-English
            df["Sentiment"] = df.apply(
                lambda row: score_to_sentiment(row["Combined_Score"]) if row["Is_English"] else rating_to_sentiment(row[rating_col]),
                axis=1
            )
            
            # Preprocess text for topic modeling (for all reviews)
            df["Processed_Tokens"] = batch_preprocess_text(df[text_col].tolist())
            df["Topic"] = batch_get_topic(df["Processed_Tokens"].tolist())
            df["Topic_Name"] = df["Topic"].map(topic_names).fillna("Unknown")

            # Filters in Sidebar
            # Month Dropdown with 'All' Option
            month_options = ["All"] + [datetime(2000, month, 1).strftime('%B') for month in range(1, 13)]
            month = st.sidebar.selectbox("Select Month", month_options)

            # Year Dropdown with 'All' Option
            year_options = ["All"] + sorted(df[date_col].apply(lambda x: x.year).unique().tolist())
            year = st.sidebar.selectbox("Select Year", year_options)

            # Company Checkboxes
            st.sidebar.subheader("🏢 Select Companies")
            company_options = sorted(df[company_col].unique().tolist())

            # Use session state to store checkbox states
            if 'company_selections' not in st.session_state:
                st.session_state.company_selections = {company: True for company in company_options}  # Default: all selected

            # Select All / Deselect All Buttons
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("Select All"):
                    st.session_state.company_selections = {company: True for company in company_options}
            with col2:
                if st.button("Deselect All"):
                    st.session_state.company_selections = {company: False for company in company_options}

            # Create a scrollable container for checkboxes
            with st.sidebar:
                with st.container(height=200):  # Adjust height as needed
                    for company in company_options:
                        st.session_state.company_selections[company] = st.checkbox(
                            company,
                            value=st.session_state.company_selections[company],
                            key=f"checkbox_{company}"
                        )

            # Get selected companies
            selected_companies = [company for company, selected in st.session_state.company_selections.items() if selected]

            # Apply Filters
            filtered_df = df.copy()

            # Filter by Companies
            if selected_companies:
                filtered_df = filtered_df[filtered_df[company_col].isin(selected_companies)]
            else:
                st.warning("⚠ Please select at least one company.")
                filtered_df = pd.DataFrame()  # Empty DataFrame if no companies are selected

            # Filter by Year and Month
            if not filtered_df.empty:
                if year != "All":
                    filtered_df = filtered_df[filtered_df[date_col].apply(lambda x: x.year) == int(year)]
                    if month != "All":
                        month_num = datetime.strptime(month, "%B").month
                        filtered_df = filtered_df[filtered_df[date_col].apply(lambda x: x.month) == month_num]

            # Main Content with Tabs
            tab1, tab2, tab3 = st.tabs(["📋 Analysis Results", "📈 Visualizations", "ℹ About"])

            with tab1:
                st.write("### 📑 Filtered Analysis Results")
                # Select columns to display
                display_columns = [company_col, "User", date_col, rating_col, text_col, "Is_English", "Rating_Score", "Model_Score", "Combined_Score", "Sentiment", "Topic_Name"]
                # Handle the case where "User" column doesn't exist
                display_columns = [col for col in display_columns if col in filtered_df.columns or col != "User"]
                if not filtered_df.empty:
                    styled_df = filtered_df[display_columns].style.applymap(highlight_sentiment, subset=['Sentiment'])
                    st.dataframe(styled_df, use_container_width=True)
                    csv = filtered_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇ Download Results as CSV", csv, "analysis_results.csv", "text/csv")
                else:
                    st.warning("No data available after applying filters.")

            with tab2:
                st.subheader("📊 Key Visualizations")
                image_files = []  # List to store image file paths

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 😊 Sentiment Distribution")
                    if not filtered_df.empty:
                        sentiment_counts = filtered_df["Sentiment"].value_counts(normalize=True) * 100
                        colors = {'positive': '#0fbf59', 'neutral': '#808080', 'negative': '#e23d2e'}
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct=lambda p: f'{p:.1f}%' if p > 0 else '', 
                            colors=[colors.get(x, '#FFFFFF') for x in sentiment_counts.index], startangle=90)
                        ax.axis('equal')
                        ax.set_title("Sentiment Proportion", fontsize=10, pad=10)
                        # Save figure
                        sentiment_image = "sentiment_distribution.png"
                        image_files.append(save_figure(fig, sentiment_image))
                        st.pyplot(fig)
                    else:
                        st.warning("No data available for sentiment distribution.")

                with col2:
                    st.markdown("### 🏢 Topic Distribution")
                    if not filtered_df.empty:
                        topic_counts = filtered_df["Topic_Name"].value_counts(normalize=True) * 100
                        topic_filter = list(topic_names.values())
                        topic_counts = topic_counts[topic_counts.index.isin(topic_filter + ["Unknown"])]
                        
                        fig, ax = plt.subplots(figsize=(6, max(4, len(topic_counts) * 0.5)))
                        y_pos = np.arange(len(topic_counts))
                        ax.barh(y_pos, topic_counts.values, color='#2EBAE2', edgecolor='black')
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(topic_counts.index, fontsize=10)
                        ax.set_xlabel('Percentage (%)', fontsize=10)
                        ax.set_title("Topic Distribution", fontsize=10, pad=10)
                        for i, v in enumerate(topic_counts.values):
                            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=8)
                        plt.tight_layout()
                        # Save figure
                        topic_image = "topic_distribution.png"
                        image_files.append(save_figure(fig, topic_image))
                        st.pyplot(fig)
                    else:
                        st.warning("No data available for topic distribution.")

                st.markdown("### 🏢 Sentiment by Company")
                if not filtered_df.empty:
                    company_sentiment = filtered_df.groupby(company_col)["Sentiment"].value_counts().unstack().fillna(0)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#e23d2e', '#808080', '#0fbf59']
                    company_sentiment.plot(kind="bar", stacked=True, color=colors, ax=ax)
                    ax.set_ylabel("Count", fontsize=10)
                    ax.legend(title="Sentiment", labels=["negative", "neutral", "positive"], fontsize=8)
                    ax.set_title("", fontsize=10)
                    plt.xticks(rotation=0, fontsize=10)
                    # Save figure
                    sentiment_company_image = "sentiment_by_company.png"
                    image_files.append(save_figure(fig, sentiment_company_image))
                    st.pyplot(fig)
                else:
                    st.warning("No data available for sentiment by company.")

                st.markdown("### 📊 Average Rating by Company")
                if not filtered_df.empty:
                    avg_rating_by_company = filtered_df.groupby(company_col)[rating_col].mean().sort_values()
                    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
                    ax.set_facecolor('white')
                    bars = ax.bar(avg_rating_by_company.index, avg_rating_by_company.values, color='#66b3ff', edgecolor='black', linewidth=0.5)
                    for bar in bars:
                        yval = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom', fontsize=10, color='black')
                    ax.set_ylim(0, 5.5)
                    ax.set_ylabel("Average Rating (1-5)", fontsize=12, color='black')
                    ax.set_title("Average Rating by Company", fontsize=14, pad=15, color='black')
                    ax.tick_params(axis='x', rotation=45, labelsize=10, colors='black')
                    ax.tick_params(axis='y', labelsize=10, colors='black')
                    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='gray')
                    ax.set_axisbelow(True)
                    plt.tight_layout()
                    # Save figure
                    rating_company_image = "average_rating_by_company.png"
                    image_files.append(save_figure(fig, rating_company_image))
                    st.pyplot(fig)
                else:
                    st.warning("No data available for average rating by company.")

                st.subheader("🧠 Bubble Chart for Topic")
                selected_topic = st.selectbox("Select Topic for Bubble Chart", list(topic_names.values()))
                adjective_only = st.checkbox("Show Adjectives Only", value=False)
                topic_id_candidates = [key for key, value in topic_names.items() if value == selected_topic]
                if topic_id_candidates:
                    topic_id = topic_id_candidates[0]
                    fig = generate_bubble_chart(topic_id, adjective_only=adjective_only)
                    if fig:
                        # Save figure
                        bubble_chart_image = "bubble_chart.png"
                        image_files.append(save_figure(fig, bubble_chart_image))
                        st.pyplot(fig)
                else:
                    st.error(f"Selected topic '{selected_topic}' not found in topic_names. Please check the topic_names dictionary.")

                st.subheader("📈 Sentiment Trend Over Time")
                if not filtered_df.empty:
                    sentiment_trend = filtered_df.groupby([date_col, "Sentiment"]).size().unstack().fillna(0)
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='black')
                    ax.set_facecolor('black')
                    sentiment_trend.plot(kind="line", ax=ax, marker="o")
                    ax.set_ylabel("Count", color='white')
                    ax.set_title("Sentiment Trend Over Time", color='white')
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')
                    ax.legend(title="Sentiment", labelcolor='white')
                    # Save figure
                    sentiment_trend_image = "sentiment_trend.png"
                    image_files.append(save_figure(fig, sentiment_trend_image))
                    st.pyplot(fig)
                else:
                    st.warning("No data available for sentiment trend.")

                # Generate and provide PDF download
                if image_files:
                    pdf_buffer, pdf_filename = generate_pdf(image_files)
                    st.download_button(
                        label="Download Visualizations as PDF",
                        data=pdf_buffer,
                        file_name=pdf_filename,
                        mime="application/pdf"
                    )
                    # Clean up image files
                    for image_file in image_files:
                        if os.path.exists(image_file):
                            os.remove(image_file)
                else:
                    st.warning("No visualizations available to include in the PDF report.")

            with tab3:
                st.write("### ℹ About")
                st.info("This dashboard performs sentiment analysis and topic modeling on reviews...")
                st.write("### Library Versions")
                st.write(f"streamlit: {st.__version__}")
                st.write(f"pandas: {pd.__version__}")
                st.write(f"torch: {torch.__version__}")
                st.write(f"transformers: {BertForSequenceClassification.__module__.split('.')[0]} {transformers.__version__}")
                st.write(f"tensorflow: {load_model.__module__.split('.')[0]} {tensorflow.__version__}")
                st.write(f"gensim: {gensim.__version__}")
                st.write(f"spacy: {spacy.__version__}")
                st.write(f"langdetect: {pkg_resources.get_distribution('langdetect').version}")
                st.write(f"matplotlib: {matplotlib.__version__}")
                st.write(f"seaborn: {sns.__version__}")
                st.write(f"numpy: {np.__version__}")
                st.write(f"requests: {requests.__version__}")
                st.info("This dashboard performs sentiment analysis and topic modeling on reviews. Upload a CSV file in the sidebar, filter by month, year, and companies using checkboxes, and explore the results.")
                st.markdown("- **Analysis Results**: View the processed data table and download the results as a CSV.")
                st.markdown("- **Visualizations**: Explore sentiment distribution, topic distribution, sentiment by company, bubble charts for topics, and sentiment trends over time.")
                st.markdown("- **Downloads**: Available in the Analysis Results tab.")