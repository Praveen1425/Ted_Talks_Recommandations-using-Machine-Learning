#!/usr/bin/env python
# coding: utf-8

# ## Ted Talks Recommendation System with Machine Learning
#
# ### Importing the Libraries and Data

import numpy as np
import pandas as pd
import nltk
import string
import warnings
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Any

# Suppress all warnings
warnings.filterwarnings('ignore')

# Download NLTK stopwords and wordnet if not already present
try:
    stopwords.words('english')
    WordNetLemmatizer().lemmatize('test')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
    nltk.download('wordnet')


def remove_stopwords(text: str) -> str:
    """
    Removes common English stopwords from a given text.

    Args:
        text (str): The input string to process.

    Returns:
        str: The string with stopwords removed.
    """
    stop_words = set(stopwords.words('english'))
    words = str(text).lower().split()
    important_words = [word for word in words if word not in stop_words]
    return " ".join(important_words)


def cleaning_punctuations(text: str) -> str:
    """
    Removes all punctuation from a given text.

    Args:
        text (str): The input string to process.

    Returns:
        str: The string with punctuation removed.
    """
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def lemmatize_text(text: str) -> str:
    """
    Lemmatizes the words in a given text.

    Args:
        text (str): The input string to process.

    Returns:
        str: The string with words lemmatized.
    """
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the raw DataFrame for the recommendation model.

    Args:
        df (pd.DataFrame): The raw DataFrame loaded from the CSV.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Combine the title and the details of the talk.
    df['combined'] = df['title'] + ' ' + df['details']

    # Remove unnecessary information
    df = df[['main_speaker', 'combined']]
    df.dropna(inplace=True)

    # Clean the text
    df['combined'] = df['combined'].apply(remove_stopwords)
    df['combined'] = df['combined'].apply(cleaning_punctuations)
    df['combined'] = df['combined'].apply(lemmatize_text)

    return df


def save_model(model_components: Dict[str, Any], file_path: str):
    """
    Saves model components to a pickle file.

    Args:
        model_components (Dict[str, Any]): Dictionary containing the model objects.
        file_path (str): The path to save the pickle file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(model_components, f)
    print(f"Model components saved to '{file_path}'")


def main():
    """
    Main function to run the TED Talks Recommendation System.
    Loads data, trains the model, and saves the components.
    """
    file_path = 'tedx_dataset.csv'
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found. Please ensure it is in the same directory.")
        return

    # Load and preprocess data
    original_df = pd.read_csv(file_path)
    processed_df = preprocess_data(original_df.copy())

    # Feature extraction with TF-IDF
    tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
    vectors = tfidf.fit_transform(processed_df['combined']).toarray()

    # Save the model components to a file
    model_components = {
        'tfidf_vectorizer': tfidf,
        'recommendation_data': processed_df # Save the processed data for later use
    }
    save_model(model_components, 'ted_talks_recommendation_model.pkl')

    print("Model training and saving complete. You can now run the Flask backend.")


if __name__ == "__main__":
    main()