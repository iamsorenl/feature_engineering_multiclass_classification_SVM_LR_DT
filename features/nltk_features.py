import nltk
from nltk import pos_tag, word_tokenize, ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

# Download required NLTK resources (if not already installed)
nltk.download('vader_lexicon')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def pos_ratios(text):
    """Calculate the ratio of nouns, verbs, and adjectives in a text."""
    print("Tokenizing text for POS ratios...")
    tokens = word_tokenize(text)          # Tokenize the text
    print("POS tagging tokens...")
    tagged = pos_tag(tokens)               # POS tagging
    print("Counting POS tags...")
    pos_counts = Counter(tag for word, tag in tagged)  # Count POS tags
    total_count = sum(pos_counts.values())
    
    # Calculate ratios for nouns, verbs, and adjectives
    pos_ratios = {
        "noun_ratio": pos_counts["NN"] / total_count if total_count else 0,
        "verb_ratio": pos_counts["VB"] / total_count if total_count else 0,
        "adj_ratio": pos_counts["JJ"] / total_count if total_count else 0
    }
    print("POS ratios calculated:", pos_ratios)
    return pos_ratios

def entity_count(text):
    """Count the number of named entities in a text."""
    print("Tokenizing text for entity count...")
    tokens = word_tokenize(text)               # Tokenize the text
    print("POS tagging for named entities...")
    tagged = pos_tag(tokens)                   # POS tagging
    print("Performing named entity chunking...")
    entities = ne_chunk(tagged)                # Named entity chunking
    entity_count = sum(1 for chunk in entities if hasattr(chunk, 'label'))  # Count entities
    print("Named entity count calculated:", entity_count)
    return {"entity_count": entity_count}

def sentiment_score(text):
    """Calculate the compound sentiment score of a text."""
    print("Calculating sentiment score...")
    sia = SentimentIntensityAnalyzer()  # Initialize the SentimentIntensityAnalyzer
    score = sia.polarity_scores(text)["compound"]  # Calculate the compound sentiment score
    print("Sentiment score calculated:", score)
    return {"sentiment": score}  # Return as dictionary

def extract_useful_features(text):
    """Extract useful features from a text."""
    print("Extracting POS tag ratios...")
    pos_features = pos_ratios(text) # Get POS Tag Ratios
    
    print("Extracting named entity count...")
    entity_features = entity_count(text) # Get Named Entity Count
    
    print("Extracting sentiment score...")
    sentiment_features = sentiment_score(text) # Get Sentiment Score
    
    features = {**pos_features, **entity_features, **sentiment_features} # Combine all features into one dictionary
    print("Extracted features:", features)
    return features
