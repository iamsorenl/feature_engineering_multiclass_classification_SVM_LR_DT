import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import time
from sklearn.model_selection import GridSearchCV
from  features.nltk_features import extract_useful_features
from joblib import Parallel, delayed

def extract_features_parallel(texts, n_jobs=-1):
    """Extract features for each text in parallel using joblib."""
    return pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(extract_useful_features)(text) for text in texts))

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

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, feature_type, model_configs):
    """Train, tune, and evaluate models using GridSearchCV with validation."""
    
    results = {}

    for name, config in model_configs.items():
        model = config['model']
        param_grid = config['param_grid']
        
        print(f"\nExploring hyperparameters for {name} with {feature_type} features...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='f1_macro',
            cv=3,
            return_train_score=True
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best parameters for {name}: {best_params}")
        
        # Validation results
        val_predictions = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_f1_score = f1_score(y_val, val_predictions, average='macro')
        
        print(f"{name} - Validation Accuracy: {val_accuracy:.4f}, Validation Macro F1: {val_f1_score:.4f}")
        
        # Test set evaluation
        test_predictions = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_f1_score = f1_score(y_test, test_predictions, average='macro')
        report = classification_report(y_test, test_predictions, target_names=list(set(y_test)))
        cm = confusion_matrix(y_test, test_predictions)
        
        results[name] = {
            'best_params': best_params,
            'validation_accuracy': val_accuracy,
            'validation_f1_score': val_f1_score,
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1_score,
            'classification_report': report,
            'confusion_matrix': cm,
            'train_time': train_time,
        }
        
        print(f"{name} - Test Accuracy: {test_accuracy:.4f}, Test Macro F1: {test_f1_score:.4f}")
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", cm)
        print("="*50)
    
    return results

def main():
    # Load the data
    data = pd.read_csv('Dataset/ecommerceDataset.csv', header=None)
    data.columns = ['Category', 'Description']
    print(data.head())

    # Convert Description column to string and handle NaN values
    data['Description'] = data['Description'].astype(str).fillna('')
    
    # Split the data into training, validation, and testing
    X_train, X_val, X_test, y_train, y_val, y_test = tt_split(data)

    # Plot the class distribution in the training and test sets
    # plot_class_distribution(y_train, y_test, y_val)

    # Combine model instances with their parameter grids
    model_configs = {
        'Logistic Regression': {
            'model': LogisticRegression(),
            'param_grid': {'C': [0.1, 1, 10], 'penalty': ['l2']}
        },
        'SVM': {
            'model': LinearSVC(),
            'param_grid': {'C': [0.1, 1, 10]}
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(),
            'param_grid': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
        }
    }
    
    # Use parallelized feature extraction on each dataset split
    print("Extracting features in parallel...")
    X_train_nltk = extract_features_parallel(X_train)
    X_val_nltk = extract_features_parallel(X_val)
    X_test_nltk = extract_features_parallel(X_test)

    feature_type = "NLTK Features"
    
    results = train_and_evaluate(X_train_nltk, X_val_nltk, X_test_nltk, y_train, y_val, y_test, feature_type, model_configs)

    # Display results
    for model, result in results.items():
        print(f"Results for {model} with {feature_type} features:")
        print(f"Best Parameters: {result['best_params']}")
        print(f"Validation Accuracy: {result['validation_accuracy']:.4f}")
        print(f"Validation Macro F1: {result['validation_f1_score']:.4f}")
        print(f"Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"Test Macro F1: {result['test_f1_score']:.4f}")
        print(f"Training Time: {result['train_time']:.4f} seconds")
        print("="*50)



if __name__ == "__main__":
    main()
