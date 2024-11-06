from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_curve, RocCurveDisplay
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import sys

def evaluate_best_model(X_train_raw, y_train, X_test_raw, y_test):
    """
    Evaluate the best Decision Tree model using OneVsRest strategy for a multi-class problem.
    This function applies Sublinear TF-IDF transformation on text data and uses a Decision Tree
    model with the optimal parameters, while plotting ROC and Precision-Recall curves.
    """

    # Transform the raw text data using Sublinear TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 1), sublinear_tf=True)
    X_train = vectorizer.fit_transform(X_train_raw).toarray()
    X_test = vectorizer.transform(X_test_raw).toarray()

    # Binarize y_train and y_test for OneVsRest classification
    lb = LabelBinarizer()
    y_train_binarized = lb.fit_transform(y_train)
    y_test_binarized = lb.transform(y_test)

    # Create the best model with determined optimal parameters
    best_model = DecisionTreeClassifier(max_depth=None, min_samples_split=2)
    ovr_classifier = OneVsRestClassifier(best_model)

    # Train the OneVsRest model on the processed training data
    ovr_classifier.fit(X_train, y_train_binarized)

    # Make predictions on the processed test data
    y_pred = ovr_classifier.predict(X_test)

    # Evaluate the predictions using accuracy and macro-average F1 score
    accuracy = accuracy_score(y_test_binarized, y_pred)
    macro_f1 = f1_score(y_test_binarized, y_pred, average='macro')

    # Print the evaluation metrics
    print(f"OneVsRest Accuracy: {accuracy:.4f}")
    print(f"OneVsRest Macro F1 Score: {macro_f1:.4f}")

    # Plot ROC curves and Precision-Recall curves for each class
    for i in range(y_test_binarized.shape[1]):
        # Compute the False Positive Rate (FPR) and True Positive Rate (TPR) for each class
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], ovr_classifier.predict_proba(X_test)[:, i])
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.title(f"ROC Curve for Class {lb.classes_[i]}")
        plt.show()

        # Compute the Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], ovr_classifier.predict_proba(X_test)[:, i])
        PrecisionRecallDisplay(precision=precision, recall=recall).plot()
        plt.title(f"Precision-Recall Curve for Class {lb.classes_[i]}")
        plt.show()

    sys.exit("Terminating process at this line.")
