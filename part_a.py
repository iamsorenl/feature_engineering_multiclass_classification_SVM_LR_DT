import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import time

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

def plot_class_distribution(y_train, y_test, y_val):
    """Plot label distribution in train, validation, and test sets."""
    # Set up three subplots for training, validation, and test distributions
    _, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    # Plot training set distribution
    y_train.value_counts().plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Training Set Distribution')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    
    # Plot validation set distribution
    y_val.value_counts().plot(kind='bar', ax=axes[1], color='lightcoral')
    axes[1].set_title('Validation Set Distribution')
    axes[1].set_xlabel('Class')
    
    # Plot test set distribution
    y_test.value_counts().plot(kind='bar', ax=axes[2], color='lightgreen')
    axes[2].set_title('Test Set Distribution')
    axes[2].set_xlabel('Class')
    
    plt.tight_layout()
    plt.show()


def main():
    # Load the data
    data = pd.read_csv('Dataset/ecommerceDataset.csv', header=None)
    data.columns = ['Category', 'Description']
    print(data.head())
    
    # Split the data into training, validation, and testing
    X_train, X_val, X_test, y_train, y_val, y_test = tt_split(data)

    # Plot the class distribution in the training and test sets
    plot_class_distribution(y_train, y_test, y_val)





if __name__ == "__main__":
    main()
