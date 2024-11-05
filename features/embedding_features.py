import numpy as np
from gensim.downloader import load as api_load

# Load GloVe embeddings using gensim API
def load_glove_embeddings(embedding_name):
    """Load GloVe embeddings using gensim API; downloads if not already cached."""
    print("Loading GloVe embeddings...")
    glove = api_load(embedding_name)
    print("GloVe embeddings loaded.")
    return glove

# Get the average word embedding for a text using gensim embeddings
def get_average_embedding(text, embeddings, embedding_dim):
    """Get the average word embedding for a text using gensim embeddings."""
    words = text.split()
    word_vectors = [embeddings[word] for word in words if word in embeddings]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(embedding_dim)  # Return zero vector if no known words found

# Apply average embedding to each text in a dataset column
def apply_average_embedding(data, column_name, embeddings, embedding_dim):
    """Apply get_average_embedding to each text in the specified column of the dataframe."""
    return np.vstack(data[column_name].apply(lambda x: get_average_embedding(x, embeddings, embedding_dim)))
