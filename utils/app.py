import os
import sys
import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from transformers import BertForSequenceClassification, BertTokenizer
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
import pickle
from datetime import datetime
import dateparser
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Add the root directory of the project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# Import Preprocessing Functions
from data_preprocessing.text_cleaning import clean_text
from data_preprocessing.tokenizer import tokenize_text
from data_preprocessing.remove_stopword import remove_stopwords
from data_preprocessing.lemmatize import lemmatize_tokens
from data_preprocessing.detect_language import detect_english, filter_english_reviews

# Date Conversion Function
def convert_relative_date(date_str):
    """Convert relative date strings like 'about a year ago' to standard YYYY-MM-DD format."""
    if not date_str or not isinstance(date_str, str):
        return None
    parsed_date = dateparser.parse(date_str)
    return parsed_date.strftime('%Y-%m-%d') if parsed_date else None

# Try to import transformers components with error handling
try:
    from transformers import BertForSequenceClassification, BertTokenizer
    print("Transformers imported successfully.")
except ImportError as e:
    st.error(f"Failed to import transformers components: {e}. Please update transformers with 'pip install --upgrade transformers' and ensure torch is installed.")
    st.stop()

# Load BERT Sentiment Model
bert_model_path = r"E:\sentiment_analysis\sentiment_model"
try:
    bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    bert_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    print("BERT Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading BERT model: {e}. Check the model path and ensure compatibility with the transformers version.")
    st.stop()

# Load BiLSTM Model and Tokenizer with detailed error handling
bilstm_model_path = r"E:\sentiment_analysis\sentiment_model\bilstm.h5"
tokenizer_path = r"E:\sentiment_analysis\sentiment_model\tokenizer.pkl"
bilstm_model = None
bilstm_tokenizer = None
try:
    if not os.path.exists(bilstm_model_path):
        raise FileNotFoundError(f"BiLSTM model file not found at {bilstm_model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
    bilstm_model = load_model(bilstm_model_path)
    with open(tokenizer_path, 'rb') as handle:
        bilstm_tokenizer = pickle.load(handle)
    print("BiLSTM Model and Tokenizer loaded successfully.")
except FileNotFoundError as e:
    st.warning(f"BiLSTM model or tokenizer not loaded: {e}. The BiLSTM option will be unavailable. Please check the file paths.")
except Exception as e:
    st.warning(f"Unexpected error loading BiLSTM model or tokenizer: {e}. The BiLSTM option will be unavailable.")

# Load LDA Model
lda_model_path = r"E:\sentiment_analysis\topic_modeling\topic_modeling/lda_model"
dictionary_path = r"E:\sentiment_analysis\topic_modeling\topic_modeling/dictionary.pkl"
corpus_path = r"E:\sentiment_analysis\topic_modeling\topic_modeling/corpus.pkl"
lda_model = LdaModel.load(lda_model_path)
dictionary = Dictionary.load(dictionary_path)
corpus = pickle.load(open(corpus_path, "rb"))

# Sentiment Mapping for Ratings
def map_rating_to_score(rating):
    """Map a rating (1-5) to a numerical sentiment score (-1 to 1)."""
    if pd.isna(rating):
        return 0  # Default to neutral if rating is missing
    rating = float(rating)
    if rating == 1:
        return -1.0  # Strongly negative
    elif rating == 2:
        return -0.5  # Moderately negative
    elif rating == 3:
        return 0.0   # Neutral
    elif rating == 4:
        return 0.5   # Moderately positive
    elif rating == 5:
        return 1.0   # Strongly positive
    return 0.0  # Default to neutral for invalid ratings

# Predict Sentiment with BERT (Return Probabilities)
def predict_sentiment_bert(text):
    if not text or pd.isna(text):
        return [0.0, 0.0, 0.0]  # Default probabilities if text is missing
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()  # Convert logits to probabilities
    return probabilities  # [P(negative), P(neutral), P(positive)]

# Predict Sentiment with BiLSTM (Return Probabilities)
def predict_sentiment_bilstm(text, max_length=100):
    if bilstm_model is None or bilstm_tokenizer is None or not text or pd.isna(text):
        return [0.0, 0.0, 0.0]  # Default probabilities if model is unavailable or text is missing
    processed_text = preprocess_text(text)
    sequences = bilstm_tokenizer.texts_to_sequences([processed_text])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    prediction = bilstm_model.predict(padded_sequences, verbose=0)
    probabilities = prediction[0].tolist()  # [P(negative), P(neutral), P(positive)]
    return probabilities

# Calculate Model Sentiment Score from Probabilities
def calculate_model_score(probabilities):
    """Convert model probabilities to a sentiment score (-1 to 1)."""
    # Sentiment values: negative = -1, neutral = 0, positive = 1
    sentiment_values = [-1, 0, 1]
    # Weighted sum: P(negative) * (-1) + P(neutral) * 0 + P(positive) * 1
    score = sum(p * v for p, v in zip(probabilities, sentiment_values))
    return score

# Combine Model and Rating Scores
def combine_scores(model_score, rating_score, model_weight=0.7, rating_weight=0.3):
    """Combine model and rating scores into a final score."""
    if model_score is None or rating_score is None:
        return 0.0  # Default to neutral if either score is missing
    combined_score = (model_weight * model_score + rating_weight * rating_score) / (model_weight + rating_weight)
    return combined_score

# Map Combined Score to Sentiment Label
def score_to_sentiment(score):
    """Map a numerical score (-1 to 1) to a sentiment label."""
    if score < -0.33:
        return "negative"
    elif score > 0.33:
        return "positive"
    else:
        return "neutral"

# Assign Topic
def get_topic(text):
    bow = dictionary.doc2bow(text.split())  # Assuming text is tokenized
    topic_probs = lda_model.get_document_topics(bow)
    return max(topic_probs, key=lambda x: x[1])[0] if topic_probs else None

# Topic Names Mapping
topic_names = {
    0: "Recommendation",
    1: "Experience",
    2: "Pricing",
    3: "Scheduling",
    4: "Environment"
}

# Apply Preprocessing
def preprocess_text(text):
    """Full text preprocessing: cleaning, tokenization, stopwords removal, and lemmatization."""
    if not text or pd.isna(text):
        return ""
    cleaned_text = clean_text(text)  # Step 1: Clean text
    tokens = tokenize_text(cleaned_text)  # Step 2: Tokenization
    filtered_tokens = remove_stopwords(tokens)  # Step 3: Remove stopwords
    lemmatized_tokens = lemmatize_tokens(filtered_tokens)  # Step 4: Lemmatization
    if isinstance(lemmatized_tokens, list):
        return " ".join(lemmatized_tokens)
    return lemmatized_tokens

# Generate Bubble Chart for a Given Topic
def generate_bubble_chart(topic_id):
    topic_terms = lda_model.show_topic(topic_id, topn=20)
    terms = [term[0] for term in topic_terms]
    weights = [term[1] for term in topic_terms]
    min_size = 700
    max_size = 2000
    word_lengths = [len(term) for term in terms]
    max_length = max(word_lengths)
    min_length = min(word_lengths)
    bubble_sizes = [
        min_size + (len(term) - min_length) / (max_length - min_length) * (max_size - min_size)
        for term in terms
    ]
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

# Streamlit UI
st.set_page_config(layout="wide")
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
                    probabilities = predict_sentiment_bert(user_input)
                elif model_option == "BiLSTM":
                    probabilities = predict_sentiment_bilstm(user_input)
                model_score = calculate_model_score(probabilities)
                rating_score = map_rating_to_score(rating_input)
                combined_score = combine_scores(model_score, rating_score)
                final_sentiment = score_to_sentiment(combined_score)
                processed_text = preprocess_text(user_input)
                topic = get_topic(processed_text)
                topic_name = topic_names.get(topic, "Unknown")
                st.success(f"**Predicted Sentiment (Combined):** {final_sentiment}")
                st.info(f"**Model Score:** {model_score:.2f}")
                st.info(f"**Rating Score:** {rating_score:.2f}")
                st.info(f"**Combined Score:** {combined_score:.2f}")
                st.info(f"**Identified Topic:** {topic_name}")
            else:
                st.warning("⚠ This text is not in English. Please enter an English review.")
        else:
            st.warning("⚠ Please enter some text.")

if option == "Upload File":
    uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.subheader("📝 Column Mapping")
        text_col = st.sidebar.selectbox("Select Review Column", df.columns)
        company_col = st.sidebar.selectbox("Select Company Column", df.columns)
        date_col = st.sidebar.selectbox("Select Date Column", df.columns)
        rating_col = st.sidebar.selectbox("Select Rating Column", df.columns)
        
        # Convert relative dates to standard format
        df[date_col] = df[date_col].apply(lambda x: convert_relative_date(str(x)))
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        median_date = df[date_col].median()
        df[date_col].fillna(median_date, inplace=True)
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Convert rating to numeric and drop rows with invalid ratings
        df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
        df = df.dropna(subset=[rating_col])
        
        # Filter for English reviews
        df = filter_english_reviews(df, text_column=text_col)
        
        # Month Dropdown with 'All' Option
        month_options = ["All"] + [datetime(2000, month, 1).strftime('%B') for month in range(1, 13)]
        month = st.sidebar.selectbox("Select Month", month_options)
        year = st.sidebar.selectbox("Select Year", df[date_col].dt.year.unique())

        if month != "All":
            month_num = datetime.strptime(month, "%B").month
            df = df[(df[date_col].dt.month == month_num) & (df[date_col].dt.year == year)]
        else:
            df = df[df[date_col].dt.year == year]
        
        # Predict sentiment probabilities based on selected model
        if model_option == "BERT":
            df["Model_Probabilities"] = df[text_col].apply(predict_sentiment_bert)
        elif model_option == "BiLSTM":
            df["Model_Probabilities"] = df[text_col].apply(predict_sentiment_bilstm)
        
        # Calculate model score
        df["Model_Score"] = df["Model_Probabilities"].apply(calculate_model_score)
        
        # Calculate rating score
        df["Rating_Score"] = df[rating_col].apply(map_rating_to_score)
        
        # Combine scores
        df["Combined_Score"] = df.apply(
            lambda row: combine_scores(row["Model_Score"], row["Rating_Score"]), axis=1
        )
        
        # Map combined score to final sentiment
        df["Sentiment"] = df["Combined_Score"].apply(score_to_sentiment)
        
        # Preprocess text for topic modeling
        df["Processed_Text"] = df[text_col].apply(preprocess_text)
        df["Topic"] = df["Processed_Text"].apply(lambda x: get_topic(x) if x else None)
        df["Topic Name"] = df["Topic"].map(topic_names)

        # Main Content with Tabs
        tab1, tab2, tab3 = st.tabs(["📋 Analysis Results", "📈 Visualizations", "ℹ About"])

        with tab1:
            st.write("### 📑 Filtered Analysis Results")
            styled_df = df.style.applymap(highlight_sentiment, subset=['Sentiment'])
            st.dataframe(styled_df, use_container_width=True)
            if not df.empty:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download Results as CSV", csv, "analysis_results.csv", "text/csv")

        with tab2:
            st.subheader("📊 Key Visualizations")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 😊 Sentiment Distribution")
                if not df.empty:
                    sentiment_counts = df["Sentiment"].value_counts(normalize=True) * 100
                    colors = {'positive': '#ADD8E6', 'neutral': '#FFA500', 'negative': '#D3D3D3'}
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct=lambda p: f'{p:.1f}%' if p > 0 else '', 
                           colors=[colors.get(x, '#FFFFFF') for x in sentiment_counts.index], startangle=90)
                    ax.axis('equal')
                    ax.set_title("Sentiment Proportion", fontsize=10, pad=10)
                    st.pyplot(fig)

            with col2:
                st.markdown("### 🏢 Topic Distribution")
                if not df.empty:
                    topic_counts = df["Topic Name"].value_counts(normalize=True) * 100
                    topic_filter = ['Experience', 'Environment', 'Recommendation']
                    topic_counts = topic_counts[topic_counts.index.isin(topic_filter)]
                    colors = {'Experience': '#ADD8E6', 'Environment': '#D3D3D3', 'Recommendation': '#FFA500'}
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(topic_counts, labels=topic_counts.index, autopct=lambda p: f'{p:.1f}%' if p > 0 else '', 
                           colors=[colors.get(x, '#FFFFFF') for x in topic_counts.index], startangle=90)
                    ax.axis('equal')
                    ax.set_title("Topic Proportion", fontsize=10, pad=10)
                    st.pyplot(fig)

            st.markdown("### 🏢 Sentiment by Company")
            if not df.empty:
                company_sentiment = df.groupby(company_col)["Sentiment"].value_counts().unstack().fillna(0)
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#FF0000', '#4682B4', '#FF4500']
                company_sentiment.plot(kind="bar", stacked=True, color=colors, ax=ax)
                ax.set_ylabel("Count", fontsize=10)
                ax.legend(title="Sentiment", labels=["negative", "neutral", "positive"], fontsize=8)
                ax.set_title("", fontsize=10)
                plt.xticks(rotation=0, fontsize=10)
                st.pyplot(fig)
            else:
                st.warning("No data available for sentiment by company.")

            st.subheader("🧠 Bubble Chart for Topic")
            selected_topic = st.selectbox("Select Topic for Bubble Chart", list(topic_names.values()))
            topic_id_candidates = [key for key, value in topic_names.items() if value == selected_topic]
            if topic_id_candidates:
                topic_id = topic_id_candidates[0]
                fig = generate_bubble_chart(topic_id)
                st.pyplot(fig)
            else:
                st.error(f"Selected topic '{selected_topic}' not found in topic_names. Please check the topic_names dictionary.")

            st.subheader("📈 Sentiment Trend Over Time")
            if not df.empty:
                sentiment_trend = df.groupby([date_col, "Sentiment"]).size().unstack().fillna(0)
                fig, ax = plt.subplots(figsize=(10, 5))
                sentiment_trend.plot(kind="line", ax=ax, marker="o")
                ax.set_ylabel("Count")
                ax.set_title("Sentiment Trend Over Time")
                st.pyplot(fig)

        with tab3:
            st.write("### ℹ About")
            st.info("This dashboard performs sentiment analysis and topic modeling on reviews. Upload a CSV file in the sidebar, filter by month and year, and explore the results.")
            st.markdown("- **Analysis Results**: View the processed data table and download the results as a CSV.")
            st.markdown("- **Visualizations**: Explore sentiment distribution, topic distribution, sentiment by company, bubble charts for topics, and sentiment trends over time.")
            st.markdown("- **Downloads**: Available in the Analysis Results tab.")

