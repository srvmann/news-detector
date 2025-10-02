import os
import yaml
import pandas as pd
import scipy.sparse
import pickle
from sklearn.linear_model import LogisticRegression
import logging

# Configure a basic logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().setLevel(logging.INFO)

# ----------------------------
# Function to load data
# ----------------------------
def load_data(data_path):
    """
    Loads vectorized features and labels for model training and evaluation.

    Args:
        data_path (str): The path to the directory containing processed data.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, and y_test,
               or (None, None, None, None) if an error occurs.
    """
    try:
        logging.info("Loading features and labels...")
        X_train = scipy.sparse.load_npz(os.path.join(data_path, "train_vectorized.npz"))
        X_test = scipy.sparse.load_npz(os.path.join(data_path, "test_vectorized.npz"))
        y_train = pd.read_csv("./data/raw/train.csv")['label']
        y_test = pd.read_csv("./data/raw/test.csv")['label']
        logging.info("Data loaded successfully.")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        logging.error("Please ensure the data preprocessing stage has been run successfully.")
        return None, None, None, None
    except KeyError:
        logging.error("Error: Please make sure your target column is named 'label' in your raw CSVs.")
        return None, None, None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}")
        return None, None, None, None

# ----------------------------
# Function to train the model
# ----------------------------
def train_model(X_train, y_train, params):
    """
    Initializes and trains a Logistic Regression model.

    Args:
        X_train (scipy.sparse.csr_matrix): Training features.
        y_train (pd.Series): Training labels.
        params (dict): Parameters for model training.

    Returns:
        sklearn.linear_model.LogisticRegression: The trained model, or None if training fails.
    """
    try:
        logging.info("Training the Logistic Regression model...")
        model = LogisticRegression(
            C = 10,
            penalty = "l2",
            solver = "liblinear",
            random_state = params.get("random_state", 42),
            max_iter = params.get("max_iter", 100)
        )
        model.fit(X_train, y_train)
        logging.info("Model training complete.")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return None

# ----------------------------
# Function to save the model
# ----------------------------
def save_model(model, model_path):
    """
    Saves the trained model to a pickle file.

    Args:
        model: The trained model object.
        model_path (str): The full path to save the model file.
    """
    try:
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            
        logging.info(f"Saving model to disk at {model_path}...")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

# ----------------------------
# Main Workflow
# ----------------------------
def main():
    """
    Main function to execute the model building pipeline.
    """
    logging.info("--- Starting Model Building Stage ---")
    
    try:
        # Load parameters
        with open("params.yaml", "r") as file:
            params = yaml.safe_load(file)
            model_params = params.get("modelbuilding", {})

        # Load data
        data_path = os.path.join("data", "processed")
        X_train, X_test, y_train, y_test = load_data(data_path)
        if X_train is None:
            logging.error("Aborting: Data loading failed.")
            return

        # Train model
        model = train_model(X_train, y_train, model_params)
        if model is None:
            logging.error("Aborting: Model training failed.")
            return

        # Save model
        model_path = os.path.join("models", "model.pkl")
        save_model(model, model_path)

    except FileNotFoundError as e:
        logging.error(f"File not found error: {e}. Please ensure params.yaml exists.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main pipeline: {e}")
        
    logging.info("--- Model Building Stage Finished ---")

if __name__ == "__main__":
    main()
