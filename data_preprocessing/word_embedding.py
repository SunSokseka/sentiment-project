import torch
import numpy as np
from transformers import BertTokenizer, BertModel

# Load BERT model and tokenizer
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BertModel.from_pretrained(bert_model_name)
model.eval()  # Set to evaluation mode (no training)

def get_bert_embedding(text):
    """Generate BERT embeddings for a given text."""
    with torch.no_grad():  # No gradient calculation (faster inference)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Extract [CLS] token representation

def apply_bert_embeddings(df, column_name="cleaned_review"):
    """Applies BERT embeddings to a given text column in a DataFrame."""
    df["embedding"] = df[column_name].apply(get_bert_embedding)
    return df

def convert_embeddings_to_numpy(df, embedding_column="embedding"):
    """Converts the embedding column to a NumPy matrix for ML models."""
    return np.vstack(df[embedding_column].values)
