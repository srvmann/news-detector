import os
import pandas as pd
import scipy.sparse
import pickle
import json
from sklearn.metrics import accuracy_score, classification_report
import logging
import mlflow
import dagshub

# Define experiment name once
EXPERIMENT_NAME = "DVC-pipeline"

# setting up dagshub
dagshub.init(repo_owner='srvmann', repo_name='news-detector', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/srvmann/news-detector.mlflow")


# Configure a basic logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().setLevel(logging.INFO)

# ----------------------------
# Function to load model and data
# ----------------------------
def load_model_and_data(model_path, test_features_path, test_labels_path):
    """
    Loads the trained model and test data from specified paths.
    """
    try:
        logging.info("Loading model and test data...")
        
        # Load the trained model from the .pkl file
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Load the test feature matrix
        X_test = scipy.sparse.load_npz(test_features_path)

        # Load the test labels from the original raw CSV
        y_test = pd.read_csv(test_labels_path)['label']
        
        logging.info("Model and test data loaded successfully.")
        return model, X_test, y_test
    except FileNotFoundError as e:
        logging.error(f"Error loading files: {e}")
        logging.error("Please ensure the model training and data processing stages have been run.")
        return None, None, None
    except KeyError:
        logging.error("Error: Ensure the target column in 'data/raw/test.csv' is named 'label'.")
        return None, None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred during file loading: {e}")
        return None, None, None

# ----------------------------
# Function to evaluate the model
# ----------------------------
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and returns key metrics.
    """
    try:
        logging.info("Making predictions on the test set...")
        y_pred = model.predict(X_test)
        
        logging.info("Calculating metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logging.info(f"\nModel Accuracy: {accuracy:.4f}")
        logging.info("Classification Report:")
        logging.info(f"\n{classification_report(y_test, y_pred)}")

        metrics_to_save = {
            'accuracy': accuracy,
            'macro_avg_precision': report['macro avg']['precision'],
            'macro_avg_recall': report['macro avg']['recall'],
            'macro_avg_f1-score': report['macro avg']['f1-score'],
            'weighted_avg_f1-score': report['weighted avg']['f1-score']
        }
        return metrics_to_save
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        return None

# ----------------------------
# Function to save metrics 
# ----------------------------
def save_metrics(metrics, metrics_path):
    """
    Saves the evaluation metrics to a JSON file.
    """
    try:
        metrics_dir = os.path.dirname(metrics_path)
        os.makedirs(metrics_dir, exist_ok=True)
        
        logging.info(f"Saving metrics to {metrics_path}...")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info("Metrics saved successfully.")
    except Exception as e:
        logging.error(f"Error saving metrics: {e}")


# ----------------------------
# Main Workflow
# ----------------------------
def main():
    """
    Main function to execute the model evaluation pipeline with MLflow tracking.
    """
    logging.info("--- Starting Model Evaluation ---")

    # 1. MLflow/DAGsHub Setup
    mlflow.set_experiment(EXPERIMENT_NAME)
    logging.info(f"MLflow experiment set to '{EXPERIMENT_NAME}'.")


    # Define paths
    model_path = os.path.join("models", "model.pkl")
    test_features_path = os.path.join("data", "processed", "test_vectorized.npz")
    test_labels_path = os.path.join("data", "raw", "test.csv")
    metrics_path = os.path.join("metrics", "eval_metrics.json")
    
    # Start MLflow run (This is the ONLY run block now)
    with mlflow.start_run(run_name="Final_Test_Evaluation_refined") as run:

        # Log basic run parameters
        mlflow.log_param("evaluation_dataset", test_labels_path)
        mlflow.log_param("model_file_used_path", model_path)
        mlflow.log_param("data_path", test_features_path)


        # Load model and data
        model, X_test, y_test = load_model_and_data(model_path, test_features_path, test_labels_path)
        if model is None or X_test is None or y_test is None:
            logging.error("Aborting: Could not load required files.")
            return

        # --- NEW: Log Model Parameters ---
        try:
            # Extract parameters using the standard scikit-learn method
            model_params = model.get_params()
            mlflow.log_params(model_params)
            logging.info("Model parameters logged to MLflow.")
        except Exception as e:
            logging.warning(f"Could not extract or log model parameters: {e}")
        # -----------------------------------


        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        if metrics is None:
            logging.error("Aborting: Model evaluation failed.")
            return
        
        # Log all calculated metrics to MLflow
        mlflow.log_metrics(metrics)
        logging.info("Evaluation metrics logged to MLflow.")

        # Save metrics locally
        save_metrics(metrics, metrics_path)
        
        # Log the generated metrics file as an artifact
        mlflow.log_artifact(metrics_path)
        logging.info("Evaluation metrics JSON logged as artifact.")
        
        # --- NEW: Log Model Artifact ---
        # Log the actual .pkl file as an artifact
        mlflow.log_artifact(model_path, artifact_path="model")
        logging.info(f"Model file '{model_path}' logged as artifact.")
        # -----------------------------------

    
    logging.info("--- Model Evaluation Finished Successfully ---")

if __name__ == "__main__":
    main()