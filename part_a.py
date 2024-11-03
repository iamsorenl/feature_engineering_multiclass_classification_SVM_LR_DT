import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

def tt_split(df):
    """Split dataset into training, validation, and test sets."""
    y = df['Category']       # Target variable
    X = df['Description']     # Feature variable
    
    # First split: 80% training/validation, 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=24, shuffle=True, stratify=y
    )
    
    # Second split: 70% training, 10% validation (relative to the original dataset)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, random_state=24, shuffle=True, stratify=y_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    # Load the data
    data = pd.read_csv('Dataset/ecommerceDataset.csv', header=None)
    data.columns = ['Category', 'Description']
    print(data.head())
    
    # Split the data into training, validation, and testing
    X_train, X_val, X_test, y_train, y_val, y_test = tt_split(data)
    
    # Print the sizes of each split
    # print("\nSplit Sizes:")
    # print(f"Training set: {X_train.shape[0]} samples")
    # print(f"Validation set: {X_val.shape[0]} samples")
    # print(f"Test set: {X_test.shape[0]} samples")
    # print("\nTarget Distribution in Each Set:")
    # print(f"Training set target distribution:\n{y_train.value_counts(normalize=True)}")
    # print(f"Validation set target distribution:\n{y_val.value_counts(normalize=True)}")
    # print(f"Test set target distribution:\n{y_test.value_counts(normalize=True)}")

    

if __name__ == "__main__":
    main()
