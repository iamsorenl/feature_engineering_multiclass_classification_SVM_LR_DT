import torch
from torchtext.vocab import GloVe
import numpy as np

# Load GloVe embeddings with caching; will only download if not already cached
def load_glove_embeddings(embedding_dim=100):
    """Load GloVe embeddings with caching; will only download if not already cached."""
    print("Loading GloVe embeddings...")
    glove = GloVe(name='6B', dim=embedding_dim)
    print("GloVe embeddings loaded.")
    return glove

# Get the average word embedding for a text
def get_average_embedding(text, embeddings, embedding_dim=100):
    """Get the average word embedding for a text."""
    words = text.split()
    word_vectors = [embeddings[word] for word in words if word in embeddings.stoi]
    if word_vectors:
        return torch.mean(torch.stack(word_vectors), dim=0).numpy()
    else:
        return np.zeros(embedding_dim)  # Return zero vector if no known words found

# Apply average embedding to each text in a dataset column
def apply_average_embedding(data, column_name, embeddings, embedding_dim=100):
    """Apply get_average_embedding to each text in the specified column of the dataframe."""
    return np.vstack(data[column_name].apply(lambda x: get_average_embedding(x, embeddings, embedding_dim)))
