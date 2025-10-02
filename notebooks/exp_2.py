import os
import re
import string
import numpy as np
import pandas as pd
import dagshub
import mlflow
import joblib
import warnings
warnings.filterwarnings('ignore')

from mlflow import log_metric, log_param, log_artifacts
from mlflow import sklearn as mlflow_sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier



dagshub.init(repo_owner='srvmann', repo_name='news-detector', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/srvmann/news-detector.mlflow")

# Laod the dataset
df = pd.read_csv(r"C:\Users\Varun\Downloads\Saurav\Outsource 360 internship\Project\Fake News Detector\News.csv")

# manual encoding the target variable (manually mapping)
df["label"] = df["label"].map({'fake':0, 'true':1})


# Function to remove all the unwanted things from our text column
def clean_text(text):
    
    text = str(text)  # Ensure it's a string

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode()

    # Convert to lowercase
    text = text.lower()

    # Remove leading/trailing and multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# new feature that holds cleaned text (free from url,email,non ascii chars or html tags)
df["clean_text"] = df["text"].apply(clean_text)
df["clean_title"] = df["title"].apply(clean_text)

# Removing the raw text and title columns
df.drop(columns=["text","title"], inplace=True)

# removing punctuation's from text and title column 
exclude = string.punctuation

# function to remove punctuation from clean_text and clean_title column
def remove_punc(text):
    return text.translate(str.maketrans("","",exclude))

# remove punctuation from clean_title column
df["clean_title"] = df["clean_title"].apply(remove_punc)

# remove punctuation from clean_text column
df["clean_text"] = df["clean_text"].apply(remove_punc) 

# Combining the text and title column and rearranging the columns
df["combined_text"] = df["clean_title"] + " " + df["clean_text"]
df = df[["combined_text","subject","date","label"]]

# Setting the experiment name
mlflow.set_experiment("Bag_of_words VS TFIDF-v01")

# Define feature extraction methods
vectorizers = {
    "Bag_of_Words": CountVectorizer(),
    "TFIDF": TfidfVectorizer()
}

# define the algorithms
alogorithms = {
    "Logistic_Regression": LogisticRegression(),
    "Naive_Bayes": MultinomialNB(),
    "Random_Forest": RandomForestClassifier(),
    "Gradient_Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier()
}

# start the parent run
import os
import joblib

with mlflow.start_run(run_name="Parent_Run") as parent_run:
    for algo_name, algorithm in alogorithms.items():
        for vec_name, vectorizer in vectorizers.items():
            
            with mlflow.start_run(run_name=f"{algo_name}_{vec_name}", nested=True):
                
                # Transform data
                X = vectorizer.fit_transform(df["combined_text"])
                y = df["label"]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, stratify=y, test_size=0.2, random_state=42
                )

                # Train model
                model = algorithm
                model.fit(X_train, y_train)

                # Evaluate
                y_pred = model.predict(X_test)
                accuracy  = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall    = recall_score(y_test, y_pred)
                f1        = f1_score(y_test, y_pred)

                # Log params & metrics
                mlflow.log_param("Vectorizer", vec_name)
                mlflow.log_param("Algorithm", algo_name)
                mlflow.log_metric("Accuracy", accuracy)
                mlflow.log_metric("Precision", precision)
                mlflow.log_metric("Recall", recall)
                mlflow.log_metric("F1_Score", f1)

               # Save the model locally using clean name
                model_filename = f"{algo_name}_{vec_name}_model.pkl"

                # Use os.path.join for better path handling
                local_path = os.path.join(os.getcwd(), model_filename)

                joblib.dump(model, local_path)

                # Log the model as an artifact 
                mlflow.log_artifact(local_path, artifact_path="models")

                # --- CLEANUP (Crucial for repeated runs) ---
                os.remove(local_path) 

                # save and log the notebook
                mlflow.log_artifact(__file__)

                # print the results for verification
                print(f"Algorithm: {algo_name}, Vectorizer: {vec_name}")
                print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
