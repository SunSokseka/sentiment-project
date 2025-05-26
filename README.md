# Sentiment Analysis Project

This project performs **sentiment analysis** and **topic modeling** on text data. It includes modules for data collection, preprocessing, sentiment modeling, topic modeling, and an interactive Streamlit-based dashboard for visualization and analysis.

---

## Project Structure

The project is organized as follows:

- **`dataset/`**: Placeholder or symlink for raw and processed datasets (not tracked in Git).
- **`notebooks/`**: Jupyter notebooks for experimentation and analysis.
  - `main.ipynb`: Primary notebook for exploratory work.
- **`src/`**: Core source code for the project.
  - **`data_annotation/`**: Scripts for data annotation.
    - `annotation.py`: Data annotation logic.
  - **`data_collection/`**: Scripts for collecting data.
    - **`data_scraping/`**: Submodule for web scraping.
      - `data_scraping.py`: Scraping script.
      - `data_scraping.ipynb`: Notebook for scraping experimentation.
  - **`data_preprocessing/`**: Data preprocessing utilities.
    - `__init__.py`: Makes this a Python package.
    - `detect_language.py`: Language detection.
    - `lemmatize.py`: Lemmatization functions.
    - `processing_text.py`: General text processing.
    - `remove_stopword.py`: Stopword removal.
    - `text_cleaning.py`: Text cleaning utilities.
    - `tokenizer.py`: Tokenization functions.
    - `word_embedding.py`: Word embedding utilities.
  - **`sentiment_models/`**: Sentiment analysis models.
    - `__init__.py`: Makes this a Python package.
    - `bert_model.py`: BERT-based sentiment model.
    - `bilstm_model.py`: BiLSTM-based sentiment model.
  - **`topic_modeling/`**: Topic modeling scripts and models.
    - `__init__.py`: Makes this a Python package.
    - **`topic_modeling_model/`**: Directory for trained topic models.
    - `dynamic_topic_naming.py`: Dynamic topic naming logic.
    - `LDA.py`: Latent Dirichlet Allocation implementation.
    - `lsa.py`: Latent Semantic Analysis implementation.
    - `NMF.py`: Non-negative Matrix Factorization implementation.
  - **`utils/`**: Utility scripts and the Streamlit app.
    - `__init__.py`: Makes this a Python package.
    - `app.py`: Streamlit dashboard script.
- **`tests/`**: Unit tests for the project.
  - `__init__.py`: Makes this a Python package.
  - `test_data_preprocessing.py`: Tests for preprocessing scripts.
  - `test_topic_modeling.py`: Tests for topic modeling scripts.
  - `test_utils.py`: Tests for utility scripts.
- **`docs/`**: Documentation files.
  - `api.md`: API documentation (if applicable).
  - `user_guide.md`: User guide for the project.
- **`.gitignore`**: Excludes unnecessary files (e.g., virtual environments, datasets).
- **`README.md`**: Project overview and setup instructions (this file).
- **`requirements.txt`**: List of Python dependencies.
- **`setup.py`**: Optional script for packaging the project (if applicable).
- **`LICENSE`**: License file (e.g., MIT, Apache 2.0).

---

## Features

- **Data Collection**: Scrape and annotate text data for analysis.
- **Preprocessing**: Clean, tokenize, lemmatize, and prepare text data.
- **Sentiment Analysis**: Use BERT and BiLSTM models to classify sentiment.
- **Topic Modeling**: Apply LDA, LSA, and NMF to identify topics in text.
- **Visualization**: Interactive Streamlit dashboard for exploring results, including sentiment distribution, topic distribution, and bubble charts.

---

