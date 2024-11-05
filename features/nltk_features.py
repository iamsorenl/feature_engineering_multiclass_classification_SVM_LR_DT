import nltk
from nltk import pos_tag, word_tokenize, ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

# Download required NLTK resources (if not already installed)
nltk.download('vader_lexicon', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def entity_count(text):
    """Count the number of named entities in a text."""
    #print("Entity Count: Starting tokenization...")
    tokens = word_tokenize(text)
    #print("Entity Count: Tokenization complete. Starting POS tagging...")
    tagged = pos_tag(tokens)
    #print("Entity Count: POS tagging complete. Starting named entity chunking...")
    entities = ne_chunk(tagged)
    entity_count_value = sum(1 for chunk in entities if hasattr(chunk, 'label'))
    #print(f"Entity Count: Named entities found: {entity_count_value}")
    return entity_count_value

def pos_ratios(text):
    """Calculate ratios of nouns, verbs, and adjectives in text."""
    #print("POS Ratios: Starting tokenization...")
    tokens = word_tokenize(text)
    #print("POS Ratios: Tokenization complete. Starting POS tagging...")
    tagged = pos_tag(tokens)
    #print("POS Ratios: POS tagging complete. Calculating tag counts...")
    pos_counts = Counter(tag for word, tag in tagged)
    total_count = sum(pos_counts.values())
    noun_ratio = pos_counts.get("NN", 0) / total_count if total_count else 0
    verb_ratio = pos_counts.get("VB", 0) / total_count if total_count else 0
    adj_ratio = pos_counts.get("JJ", 0) / total_count if total_count else 0
    #print(f"POS Ratios calculated: Noun Ratio: {noun_ratio}, Verb Ratio: {verb_ratio}, Adjective Ratio: {adj_ratio}")
    return {
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "adj_ratio": adj_ratio
    }

def sentiment_score(text):
    """Calculate the sentiment score of a text."""
    #print("Sentiment Score: Initializing sentiment analysis...")
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)["compound"]
    #print(f"Sentiment Score calculated: {score}")
    return score

def extract_useful_features(text):
    """Extract various NLP-based features from a text."""
    #print("\nStarting feature extraction for a new text entry...")
    features = {}
    
    #print("Extracting entity count...")
    features["entity_count"] = entity_count(text)
    #print("Entity count extraction complete.")
    
    #print("Extracting POS ratios...")
    features.update(pos_ratios(text))
    #print("POS ratios extraction complete.")
    
    #print("Extracting sentiment score...")
    features["sentiment"] = sentiment_score(text)
    #print("Sentiment score extraction complete.")
    
    #print("Calculating word and character counts...")
    features["word_count"] = len(word_tokenize(text))
    features["char_count"] = len(text)
    #print(f"Word count: {features['word_count']}, Character count: {features['char_count']}")
    
    #print("Feature extraction completed for the text entry.")
    return features
