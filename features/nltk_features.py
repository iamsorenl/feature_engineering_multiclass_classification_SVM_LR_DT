from nltk import pos_tag, word_tokenize, ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

def pos_ratios(text):
    """Calculate the ratio of nouns, verbs, and adjectives in a text."""
    tokens = word_tokenize(text)          # Tokenize the text
    tagged = pos_tag(tokens)               # POS tagging
    pos_counts = Counter(tag for word, tag in tagged)  # Count POS tags
    total_count = sum(pos_counts.values())
    
    # Calculate ratios for nouns, verbs, and adjectives
    pos_ratios = {
        "noun_ratio": pos_counts["NN"] / total_count if total_count else 0,
        "verb_ratio": pos_counts["VB"] / total_count if total_count else 0,
        "adj_ratio": pos_counts["JJ"] / total_count if total_count else 0
    }
    return pos_ratios

def entity_count(text):
    """Count the number of named entities in a text."""
    tokens = word_tokenize(text)               # Tokenize the text
    tagged = pos_tag(tokens)                   # POS tagging
    entities = ne_chunk(tagged)                # Named entity chunking
    entity_count = sum(1 for chunk in entities if hasattr(chunk, 'label'))  # Count entities
    return {"entity_count": entity_count}

def sentiment_score(text):
    """Calculate the compound sentiment score of a text."""
    sia = SentimentIntensityAnalyzer()  # Initialize the SentimentIntensityAnalyzer
    score = sia.polarity_scores(text)["compound"]  # Calculate the compound sentiment score
    return score

def extract_useful_features(text):
    """Extract useful features from a text."""
    pos_features = pos_ratios(text) # Get POS Tag Ratios
    
    entity_features = entity_count(text) # Get Named Entity Count
    
    sentiment_features = sentiment_score(text) # Get Sentiment Score
    
    features = {**pos_features, **entity_features, **sentiment_features} # Combine all features into one dictionary
    return features
