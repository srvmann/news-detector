from flask import Flask, render_template, request
from pre_processing import preprocess_text 
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import os
import pickle
import logging
import sys

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- CONFIGURATION ---
MLFLOW_TRACKING_URI = "http://ec2-13-53-108-155.eu-north-1.compute.amazonaws.com:5000/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# 2. DEFINE MODEL PATHS
MODEL_NAME = "News_Detector_LR_BOW"
MODEL_VERSION = 1  

# --- NEW: LOCAL PATH FOR MANUALLY SUPPLIED VECTORIZER ---
# Define the file path as a string (assumes 'vectorizer.pkl' is in the same directory)
LOCAL_VECTORIZER_PATH = "vectorizer.pkl"
# --------------------------------------------------------


app = Flask(__name__)

# --- GLOBAL VARIABLES ---
# Initialize globally, but load inside load_resources
model = None
vectorizer = None


# --- MODEL AND ARTIFACT LOADING ---

def load_resources():
    """Loads the registered model (LR) from MLflow and attempts to load the vectorizer locally."""
    global model, vectorizer
    
    # 1. Load Model from MLflow Registry
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        # Load the model into the global variable
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info(f"✅ Loaded model from MLflow Registry: {model_uri}")

    except Exception as e:
        logging.error(f"FATAL ERROR LOADING MLFLOW MODEL: {e}")
        logging.error("Check MLflow server availability, tracking URI, and model name/version.")
        # Exit if the critical resource (the model) cannot be loaded
        sys.exit(1)
        
    # 2. Load Vectorizer Manually (Since it wasn't logged correctly)
    try:
        if os.path.exists(LOCAL_VECTORIZER_PATH):
            with open(LOCAL_VECTORIZER_PATH, 'rb') as f:
                # Load the vectorizer object into the global variable
                vectorizer = pickle.load(f)
            logging.info(f"✅ Loaded vectorizer successfully from local file: {LOCAL_VECTORIZER_PATH}")
        else:
            vectorizer = None
            logging.critical(f"⚠️ CRITICAL WARNING: Vectorizer file not found at '{LOCAL_VECTORIZER_PATH}'.")
            logging.critical("Prediction will FAIL.")
            # Note: We do NOT sys.exit(1) here, allowing the app to run with the warning.

    except Exception as e:
        logging.error(f"FATAL ERROR LOADING LOCAL VECTORIZER: {e}")
        sys.exit(1)


# Run resource loading once at startup
load_resources()


# --- PREPROCESSING FUNCTION (Assuming this is a placeholder for your imported module) ---
# NOTE: If 'from pre_processing import preprocess_text' is a valid import, 
# you should remove this definition. Keeping it here for safety against import errors.
def preprocess_text(text):
    """Placeholder for the imported text cleaning function."""
    import re
    import string
    
    # Combine title and text cleaning steps from your training pipeline:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text
# -------------------------------------------------------------------------------------


@app.route('/')
def home():
    # Model/Vectorizer check is redundant here but safe
    return render_template('index.html', result=None)


@app.route('/predict', methods=['POST'])
def predict():
    global vectorizer # Access global vectorizer state 
    global model      # Access global model state

    raw_text = request.form['text']
    
    # 1. Check if vectorizer is available
    if vectorizer is None:
        logging.error("Prediction attempt failed: Vectorizer not loaded.")
        return render_template('index.html', 
                               input_text=raw_text,
                               result=f"ERROR: Preprocessing module (Vectorizer) missing at {LOCAL_VECTORIZER_PATH}.",
                               result_class="text-red-600 font-bold")

    # 2. Clean (Preprocess)
    clean_text = preprocess_text(raw_text)

    # 3. Vectorization (CRITICAL STEP)
    try:
        # Vectorizer expects an iterable (list) of strings
        input_vector = vectorizer.transform([clean_text])
    except Exception as e:
        logging.error(f"Vectorization failed: {e}")
        return render_template('index.html', 
                               input_text=raw_text,
                               result="ERROR: Vectorization failed.",
                               result_class="text-red-600 font-bold")

    # 4. Prediction
    # Use the global model variable
    prediction = model.predict(input_vector)[0]
    
    # 5. Format Result
    result_text = "FAKE News" if prediction == 0 else "TRUE News"
    result_class = "text-red-600 font-bold" if prediction == 0 else "text-green-600 font-bold"

    return render_template('index.html', 
                           input_text=raw_text,
                           result=result_text,
                           result_class=result_class)

if __name__ == '__main__':
    app.run(debug=True)
