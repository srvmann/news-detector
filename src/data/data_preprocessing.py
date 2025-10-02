import os
import re
import string
import yaml
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse
import logging

# Configure a basic logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().setLevel(logging.INFO)

# ----------------------------
# Text Preprocessing Functions
# ----------------------------

def preprocess_text(text):
    """
    Cleans and preprocesses a single string of text.

    Steps include:
    - Converting to lowercase.
    - Removing HTML tags, URLs, and emails.
    - Removing punctuation.
    - Removing non-ASCII characters.
    - Tokenizing and removing stopwords.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The cleaned and preprocessed text.
    """
    # Ensure it's a string and convert to lowercase
    text = str(text).lower()
    
    # Remove HTML, URLs, and emails
    text = re.sub(r"<.*?>|http\S+|www\.\S+|\S+@\S+", "", text)
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode()
    
    # Tokenize and remove stopwords
    words = text.split()
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Join back into a single string
    return " ".join(filtered_words)

# ----------------------------
# Data Processing and Vectorization
# ----------------------------

def process_and_vectorize_data(train_df, test_df, params):
    """
    Applies text preprocessing and vectorization to the dataframes.

    Args:
        train_df (pd.DataFrame): The training dataframe.
        test_df (pd.DataFrame): The testing dataframe.
        params (dict): Dictionary of parameters from params.yaml.

    Returns:
        tuple: A tuple containing:
            - A processed train dataframe with a 'combined_text' column.
            - A processed test dataframe with a 'combined_text' column.
            - The vectorized training data (sparse matrix).
            - The vectorized testing data (sparse matrix).
            - The fitted CountVectorizer object.
    """
    try:
        logging.info("Starting text preprocessing for train and test data...")
        # Download NLTK stopwords
        nltk.download("stopwords", quiet=True)

        # Apply the preprocessing function
        train_df['title'] = train_df['title'].apply(preprocess_text)
        train_df['text'] = train_df['text'].apply(preprocess_text)
        test_df['title'] = test_df['title'].apply(preprocess_text)
        test_df['text'] = test_df['text'].apply(preprocess_text)

        # Combine the cleaned title and text columns
        train_df['combined_text'] = train_df['title'] + ' ' + train_df['text']
        test_df['combined_text'] = test_df['title'] + ' ' + test_df['text']
        logging.info("Text preprocessing complete.")
        
        logging.info("Starting text vectorization with CountVectorizer...")
        max_features = params.get("max_features")
        min_df = params.get("min_df")
        
        # Initialize the CountVectorizer
        vectorizer = CountVectorizer(max_features=max_features, min_df=min_df)

        # Fit on training data and transform both train and test data
        X_train_vectorized = vectorizer.fit_transform(train_df['combined_text'])
        X_test_vectorized = vectorizer.transform(test_df['combined_text'])
        
        logging.info("Text vectorization complete.")
        return train_df, test_df, X_train_vectorized, X_test_vectorized, vectorizer

    except Exception as e:
        logging.error(f"Error during data processing and vectorization: {e}")
        return None, None, None, None, None

# ----------------------------
# Save Processed Data
# ----------------------------

def save_processed_data(data_path, train_df, test_df, X_train_vectorized, X_test_vectorized):
    """
    Saves processed dataframes and vectorized matrices.

    Args:
        data_path (str): The directory path for saving files.
        train_df (pd.DataFrame): Processed training dataframe.
        test_df (pd.DataFrame): Processed testing dataframe.
        X_train_vectorized (scipy.sparse.csr_matrix): Vectorized training data.
        X_test_vectorized (scipy.sparse.csr_matrix): Vectorized testing data.
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(data_path, exist_ok=True)
        logging.info(f"Saving processed files to {data_path}")

        # Save the dataframes
        train_df.to_csv(os.path.join(data_path, "train_processed_text.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_processed_text.csv"), index=False)

        # Save the vectorized sparse matrices
        scipy.sparse.save_npz(os.path.join(data_path, "train_vectorized.npz"), X_train_vectorized)
        scipy.sparse.save_npz(os.path.join(data_path, "test_vectorized.npz"), X_test_vectorized)
        logging.info("All processed files saved successfully.")

    except Exception as e:
        logging.error(f"Error saving processed data: {e}")

# ----------------------------
# Main Workflow
# ----------------------------

def main():
    """
    Main function to execute the data preprocessing and vectorization pipeline.
    """
    try:
        # Load parameters
        with open("params.yaml", "r") as file:
            params = yaml.safe_load(file)
            data_preprocessing_params = params.get("dataPreprocessing", {})
        
        # Load dataframes
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logging.info(f"Loaded train data (shape: {train_data.shape}) and test data (shape: {test_data.shape}).")

        # Process and vectorize data
        processed_train, processed_test, X_train, X_test, vectorizer = process_and_vectorize_data(
            train_data.copy(), test_data.copy(), data_preprocessing_params
        )

        if processed_train is None or processed_test is None:
            logging.error("Aborting: Data processing failed.")
            return

        # Define save path and save the processed files
        processed_data_path = os.path.join("data", "processed")
        save_processed_data(processed_data_path, processed_train, processed_test, X_train, X_test)

    except FileNotFoundError as e:
        logging.error(f"File not found error: {e}. Please ensure params.yaml and data files exist.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main pipeline: {e}")

if __name__ == "__main__":
    main()