# Text Mining and Sentiment Analysis Project

## Overview

This project focuses on text mining and sentiment analysis using machine learning and deep learning techniques. It preprocesses textual data, extracts features, and builds models to classify sentiment scores based on user reviews. The project includes traditional machine learning approaches (e.g., Naive Bayes, K-Nearest Neighbors) as well as advanced deep learning models like Bidirectional LSTMs.

---

## Features

### 1. **Dataset**
- The dataset contains user reviews and corresponding sentiment scores.
- Columns include:
  - `Score`: Sentiment score (e.g., 1–5).
  - `Text`: User review text.

### 2. **Text Preprocessing**
- Tokenization using `nltk.word_tokenize`.
- Removal of stopwords and punctuation.
- Lemmatization using `WordNetLemmatizer`.
- Part-of-speech tagging for better context understanding.

### 3. **Feature Extraction**
- **TF-IDF Vectorization**: Converts text into numerical features based on term frequency-inverse document frequency.
- **Embedding Layers**: Used in deep learning models to represent words as dense vectors.

### 4. **Models**
#### **Machine Learning Models**
- **Naive Bayes**: A probabilistic classifier based on word frequencies.
- **K-Nearest Neighbors (KNN)**: Classifies text based on similarity to neighboring data points.

#### **Deep Learning Models**
- **Bidirectional LSTM**: Captures context in both forward and backward directions for improved sentiment classification.
- **CNN**: Extracts features from sequences using convolutional layers for sentiment prediction.

### 5. **Model Evaluation**
Metrics used for evaluation include:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Dependencies

### Required Python Libraries:
```python
pip install pandas numpy matplotlib seaborn nltk sklearn tensorflow imbalanced-learn
```

---

## How to Run

### Steps:
1. Clone or download the project repository.
2. Place the dataset (`dataset.csv`) in the appropriate directory specified in the notebook.
3. Install required libraries using the command above.
4. Open the notebook `Text_Mining_ASM.ipynb` in Jupyter Notebook or any compatible IDE.
5. Execute each cell sequentially to preprocess data, train models, and evaluate results.

---

## Outputs

### Visualizations:
- Word clouds showing frequent terms in positive and negative reviews.
- Bar plots displaying sentiment score distributions.
- Confusion matrices for model evaluation.

### Model Results:
Performance metrics for each model are displayed, including accuracy, precision, recall, and F1 score.

---

## Contact
For questions or suggestions regarding this project, feel free to reach out!

