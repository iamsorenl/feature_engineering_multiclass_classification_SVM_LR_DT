# Feature Engineering for SVM, Logistic Regression & Decision Tree

This project applies feature engineering techniques to **SVM**, **Logistic Regression**, and **Decision Tree** classifiers to solve a 4-class classification problem using an e-commerce dataset. Various feature extraction methods, including **GloVe embeddings**, **Sublinear TF-IDF**, and a **combination of NLTK and Bag-of-Words features**, are used to optimize model performance.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Feature Engineering](#feature-engineering)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Timing Comparison](#timing-comparison)
- [Hyperparameter Exploration](#hyperparameter-exploration)
- [OneVsRest Exploration](#onevsrest-exploration)
- [Contributing](#contributing)

---

## Project Overview

This project focuses on classifying e-commerce item descriptions into four categories using **Logistic Regression**, **SVM**, and **Decision Tree** classifiers from **scikit-learn**. The goal is to explore feature engineering methods to maximize classification accuracy and evaluate the models using multiple metrics, including **accuracy** and **macro-average F1 score**.

---

## Requirements

**Main Dependencies:**
- Python >= 3.11
- scikit-learn
- pandas
- numpy
- nltk
- Gensim (for GloVe embeddings)

Install all required dependencies using the command:

```bash
pip install -r requirements.txt
```

---

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/feature-engineering-svm-lr-dt.git
    cd feature-engineering-svm-lr-dt
    ```

2. **Create a Virtual Environment**

    To avoid conflicts with other packages, use a virtual environment:

    ```bash
    python3 -m venv venv
    # On MacOS/Linux
    source venv/bin/activate
    # On Windows
    venv\Scripts\activate
    ```

3. **Install Dependencies**

    Install all required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

---

## Dataset Structure

Ensure the following CSV files are present in the project directory:

- `ecommerceDataset.csv`: Contains item categories and descriptions for training and testing.

Each dataset follows this structure:

**Training Dataset (`ecommerceDataset.csv`):**

| ID | CATEGORY | DESCRIPTION           |
|----|-------------|--------------------|
| 1  | Books       | Lorem ipsum        |
| 2  | ...         | ...

---

## Usage

### Training and Evaluating the Models

To train and evaluate the models, run:

```bash
python run.py ecommerceDataset.csv
```

This command will:

- Split the dataset into training, validation, and test sets.
- Apply feature engineering techniques.
- Train and evaluate models using **Logistic Regression**, **SVM**, and **Decision Tree** classifiers.

---

## Feature Engineering

The following feature engineering techniques were applied:

- **GloVe Embeddings**: Average word embeddings were generated using pre-trained `glove-wiki-gigaword-300` model from Gensim.
- **Sublinear TF-IDF**: Term frequency-inverse document frequency with sublinear scaling to emphasize less frequent terms.
- **Combined NLTK and Bag-of-Words Features**:
  - Named Entity Count (using NLTK)
  - Part-of-Speech Ratios (nouns, verbs, adjectives)
  - Sentiment Score (using NLTK’s SentimentIntensityAnalyzer)
  - Basic Counts (word and character counts)

---

## Model Training and Evaluation

The models were trained using **scikit-learn's** `train_test_split` and evaluated on their accuracy and macro-average F1 scores. Cross-validation and hyperparameter tuning were conducted to identify optimal settings.

### Metrics Reported

- **Accuracy**: Overall correctness of predictions.
- **Macro F1 Score**: Balances precision and recall across multiple classes.

---

## Timing Comparison

Timing data was recorded for each feature extraction and model training process to compare computational efficiency.

### Example Timing Summary

| Feature Set                  | Extraction Time | Logistic Regression | SVM                | Decision Tree       |
|-----------------------------|-----------------|---------------------|-------------------|---------------------|
| GloVe Embeddings            | 27.68 sec       | 20.70 sec           | 1 min 14.06 sec   | 29.51 sec           |
| Sublinear TF-IDF            | 1.49 sec        | 11.68 sec           | 2.84 sec          | 27.93 sec           |
| Combined NLTK + Bag-of-Words| 1.48 sec + ...  | 30.07 sec           | 22 min 51.08 sec  | 21.76 sec           |

---

## Hyperparameter Exploration

Key hyperparameters were explored for each model:

- **C (Regularization)**: Controls regularization strength in Logistic Regression and SVM.
- **max_depth** and **min_samples_split**: For controlling Decision Tree complexity.

---

## OneVsRest Exploration

A OneVsRest approach was used to evaluate the multi-class problem, providing individual accuracy and macro-average F1 scores for each class. **ROC** and **Precision-Recall curves** were generated for each class to assess discriminative performance.

---

## Contributing

Contributions are welcome! If you encounter issues or have ideas for improvements, feel free to open a pull request or issue.

---
