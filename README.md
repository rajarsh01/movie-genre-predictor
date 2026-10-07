# 🎬 AI Movie Genre Predictor

A Machine Learning and Natural Language Processing (NLP) web application that predicts a movie's genre based entirely on its plot summary. 


## 🧠 Project Overview
This project utilizes a custom-trained Logistic Regression model and TF-IDF vectorization to analyze narrative patterns and vocabulary in movie plots. It was trained on a dataset of thousands of movie summaries across 27 distinct genres. To improve accuracy on rare genres and complex prompts, the model utilizes balanced class weights and bi-gram (2-word phrase) recognition.

## ✨ Key Features
*   **Instant Predictions:** Analyzes text and returns the most likely genre in milliseconds.
*   **Interactive Confidence Chart:** Visualizes the top 5 predicted genres and their respective probability scores using Altair.
*   **Explainable AI (NLP Insights):** Reverses the TF-IDF vectorization to extract and display the exact words and phrases from the user's input that most heavily influenced the model's decision.
*   **Dynamic Text Processing:** Automatically cleans user input by removing punctuation, standardizing casing, and filtering out English stopwords via NLTK.

## 🛠️ Technology Stack
*   **Frontend & Hosting:** Streamlit, Streamlit Community Cloud
*   **Machine Learning:** Scikit-learn (Logistic Regression, TF-IDF Vectorizer)
*   **Natural Language Processing:** NLTK (Stopwords)
*   **Data Manipulation & Visualization:** Pandas, NumPy, Altair
*   **Language:** Python 3

## 💻 How to Run Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/rajarsh01/movie-genre-predictor.git](https://github.com/rajarsh01/movie-genre-predictor.git)

   cd movie-genre-predictor
   pip install -r requirements.txt
   streamlit run app.py
