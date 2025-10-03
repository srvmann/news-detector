# register model

import json
import mlflow
import logging
import os
import mlflow.sklearn
# Removed unused imports: from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --- Configuration ---
# Set the Tracking URI to your self-hosted MLflow server (AWS EC2)
mlflow.set_tracking_uri("http://ec2-13-53-108-155.eu-north-1.compute.amazonaws.com:5000/")

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

    logger.info(f"Model URI: {model_uri}")
    logger.info(f"Model Name: {model_name}")

    # --- Full registry flow (Now guaranteed to run against the AWS endpoint) ---
    try:
        model_version = mlflow.register_model(model_uri, model_name)
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.info(f"✅ Model {model_name} version {model_version.version} registered and moved to Staging.")
        return model_version
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        # This will now catch any non-permission errors (like network/database issues)
        raise 


def main():
    """Main function to orchestrate model registration."""
    
    try:
        # This JSON file contains the Run ID and the artifact path ("model")
        model_info_path = 'reports/model_evaluation_info.json' 
        
        model_info = load_model_info(model_info_path)
        
        model_name = "News_Detector_LR_BOW" 
        
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()