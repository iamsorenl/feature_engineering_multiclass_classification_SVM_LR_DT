import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
import time
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from features.nltk_features import extract_useful_features
from features.embedding_features import load_glove_embeddings, apply_average_embedding
import warnings
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
import sys # For sys.exit() function
from onevsrest import evaluate_best_model

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def extract_features_parallel(texts, n_jobs=-1):
    """Extract features for each text in parallel using joblib."""
    return pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(extract_useful_features)(text) for text in texts['Description']))

def tt_split(df):
    """Split dataset into training, validation, and test sets."""
    y = df['Category']       # Target variable
    X = df[['Description']]     # Feature variable
    
    # First split: 80% training/validation, 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=24, shuffle=True, stratify=y
    )
    
    # Second split: 70% training, 10% validation (relative to the original dataset)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, random_state=24, shuffle=True, stratify=y_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, feature_type, model_configs):
    """Train, tune, and evaluate models using GridSearchCV with validation."""
    
    for name, config in model_configs.items():
        model = config['model']
        param_grid = config['param_grid']
        
        print(f"\n{'='*10} Exploring {name} with {feature_type} features {'='*10}\n")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='f1_macro',
            cv=3,
            n_jobs=-1,
            return_train_score=True
        )

        print("Finished grid search setup. Starting training...\n")
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Print detailed performance of each parameter combination across validation folds
        print("\nDetailed Validation Performance for Each Parameter Combination:\n")
        for idx, params in enumerate(grid_search.cv_results_['params']):
            print(f"Parameters: {params}")
            # Get F1 Macro Score for each fold for the current parameter combination
            fold_scores = [
                grid_search.cv_results_[f'split{fold}_test_score'][idx]
                for fold in range(grid_search.cv)
            ]
            for fold, score in enumerate(fold_scores):
                print(f"    Fold {fold + 1}: F1 Macro Score: {score:.4f}")
            mean_score = grid_search.cv_results_['mean_test_score'][idx]
            std_score = grid_search.cv_results_['std_test_score'][idx]
            print(f"    Mean F1 Macro Score: {mean_score:.4f} (+/- {std_score:.4f})\n")

        # Show best parameters found and training time
        print(f"Best Parameters for {name}:")
        for param, value in best_params.items():
            print(f"    {param}: {value}")
        print(f"Training Time: {train_time:.2f} seconds\n")
        
        # Validation results with best parameters
        val_predictions = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_f1_score = f1_score(y_val, val_predictions, average='macro', zero_division=0)
        
        print(f"Validation Results for {name}:")
        print(f"    Accuracy: {val_accuracy:.4f}")
        print(f"    Macro F1: {val_f1_score:.4f}\n")
        
        # Test set evaluation with best parameters
        test_predictions = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_f1_score = f1_score(y_test, test_predictions, average='macro', zero_division=0)
        report = classification_report(y_test, test_predictions, target_names=list(set(y_test)))
        
        # Displaying the confusion matrix visually using ConfusionMatrixDisplay
        cm = confusion_matrix(y_test, test_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(set(y_test)))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for {name} with {feature_type}')
        plt.show()
        
        print(f"Test Results for {name}:")
        print(f"    Accuracy: {test_accuracy:.4f}")
        print(f"    Macro F1: {test_f1_score:.4f}\n")
        print("Classification Report:\n", report)
        print(f"{'='*50}\n")

def main():
    # Load the data
    data = pd.read_csv('Dataset/ecommerceDataset.csv', header=None)
    data.columns = ['Category', 'Description']
    print(data.head())

    # Convert Description column to string and handle NaN values
    data['Description'] = data['Description'].astype(str).fillna('')
    
    # Split the data into training, validation, and testing
    X_train, X_val, X_test, y_train, y_val, y_test = tt_split(data)

    # uncomment the following line to run the OneVsRestClassifier with Decision Tree and Sublinear TF-IDF
    # This model combo was deterined to be the best after running through several combinations
    # evaluate_best_model(X_train['Description'], y_train, X_test['Description'], y_test)

    # Combine model instances with their parameter grids
    model_configs = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=5000, class_weight='balanced'),
            'param_grid': {'C': [0.01, 0.1, 1, 10, 100]}
        },
        'SVM': {
            'model': LinearSVC(max_iter=10000, class_weight='balanced'),
            'param_grid': {'C': [0.01, 0.1, 1, 10, 100]}
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(class_weight='balanced'),
            'param_grid': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
        }
    }

    # GloVe Embedding Loading and Feature Extraction
    print("Starting GloVe embedding loading...")
    start_time = time.time()

    glove_embeddings = load_glove_embeddings(embedding_name="glove-wiki-gigaword-300")
    embedding_dim = glove_embeddings.vector_size  # Adjust if different dimensions are used

    X_train_glove = apply_average_embedding(X_train, 'Description', glove_embeddings, embedding_dim)
    X_val_glove = apply_average_embedding(X_val, 'Description', glove_embeddings, embedding_dim)
    X_test_glove = apply_average_embedding(X_test, 'Description', glove_embeddings, embedding_dim)

    end_time = time.time()
    print(f"GloVe embedding loading and feature extraction completed in {end_time - start_time:.2f} seconds.\n")

    feature_type = "GloVe Embeddings"
    train_and_evaluate(X_train_glove, X_val_glove, X_test_glove, y_train, y_val, y_test, feature_type, model_configs)

    # Uncomment the following line to terminate the process after the first feature extraction
    # sys.exit("Terminating process at this line.")

    # Sublinear TF-IDF Feature Extraction
    print("Starting sublinear TF-IDF extraction...")
    start_time = time.time()

    sublinear_tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 1), sublinear_tf=True)
    X_train_tfidf = sublinear_tfidf.fit_transform(X_train['Description']).toarray()
    X_val_tfidf = sublinear_tfidf.transform(X_val['Description']).toarray()
    X_test_tfidf = sublinear_tfidf.transform(X_test['Description']).toarray()

    end_time = time.time()
    print(f"Sublinear TF-IDF extraction completed in {end_time - start_time:.2f} seconds.\n")

    feature_type = "Sublinear TF-IDF Features"
    train_and_evaluate(X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test, feature_type, model_configs)

    # Uncomment the following line to terminate the process after the second feature extraction
    # sys.exit("Terminating process at this line.")

    # Bag-of-Words Feature Extraction
    print("Starting Bag-of-Words extraction...")
    start_time = time.time()

    bow_vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 1))
    X_train_bow = bow_vectorizer.fit_transform(X_train['Description']).toarray()
    X_val_bow = bow_vectorizer.transform(X_val['Description']).toarray()
    X_test_bow = bow_vectorizer.transform(X_test['Description']).toarray()

    end_time = time.time()
    print(f"Bag-of-Words extraction completed in {end_time - start_time:.2f} seconds.\n")
    
    # NLTK Feature Extraction
    print("Starting parallel feature extraction using NLTK...")
    start_time = time.time()

    X_train_nltk = extract_features_parallel(X_train)
    X_val_nltk = extract_features_parallel(X_val)
    X_test_nltk = extract_features_parallel(X_test)

    end_time = time.time()
    print(f"Parallel NLTK feature extraction completed in {end_time - start_time:.2f} seconds.\n")

    # Concatenate NLTK features and Bag-of-Words
    X_train_combined = np.hstack((X_train_nltk, X_train_bow))
    X_val_combined = np.hstack((X_val_nltk, X_val_bow))
    X_test_combined = np.hstack((X_test_nltk, X_test_bow))

    # Scaling Features
    print("Scaling features...")
    start_time = time.time()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_val_scaled = scaler.transform(X_val_combined)
    X_test_scaled = scaler.transform(X_test_combined)

    end_time = time.time()
    print(f"Feature scaling completed in {end_time - start_time:.2f} seconds.\n")

    feature_type = "Combined NLTK and Bag-Of-Words Features"
    train_and_evaluate(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_type, model_configs)

if __name__ == "__main__":
    main()
