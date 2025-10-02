# Hyper-parameter tunning the model which gave best results in exp_2.py
# [Logistic Regresion + Bag of words]

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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV



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

# Initializing the CountVectorizer
vectorizer = CountVectorizer()

# Transforming the text data to feature vectors that can be used as input to the model
X = vectorizer.fit_transform(df["combined_text"])
y = df["label"]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.2, random_state=42)

# Setting the experiment name
mlflow.set_experiment("Logistic Regression Hyperparameter Tunning-v01")

# Hyperparameter tuning for Logistic Regression
param__grid = {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'saga'],
                'penalty': ['l1', 'l2']
}   

# start the parent run for hyper-parameter tuning
with mlflow.start_run(run_name="Logistic_Regression_Hyperparameter_Tunning"):

    # perform grid search
    grid_search = GridSearchCV(LogisticRegression(), param_grid = param__grid, cv = 5, scoring = 'f1', n_jobs = -1)
    grid_search.fit(X_train, y_train)

    # log each parameter combination as child run
    for params, mean_score, std_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score']):
        with mlflow.start_run(run_name = f"LR with params_{params}", nested = True):
                   
            # train model with given parameters
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)
            
            # make predictions
            y_pred = model.predict(X_test)
            
            # calculate metrics
            accuracy  = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall    = recall_score(y_test, y_pred)
            f1        = f1_score(y_test, y_pred)
            
            # log metrics
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_f1_score", mean_score)
            mlflow.log_metric("std_cv_f1_score", std_score)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)


    # Log the best model from grid search
    best_params = grid_search.best_params_
    best_score  = grid_search.best_score_
    mlflow.log_params(best_params)
    mlflow.log_metric("best_f1_score", best_score)  
            
    # save and log the notebook
    mlflow.log_artifact(__file__)

    # log model 
    # Save the model locally using clean name
    model_filename = "LogisticRegression_BOW_model.pkl"

    # Use os.path.join for better path handling
    local_path = os.path.join(os.getcwd(), model_filename)

    joblib.dump(grid_search.best_estimator_, local_path)

    # Log the model as an artifact 
    mlflow.log_artifact(local_path, artifact_path="models")
            
           