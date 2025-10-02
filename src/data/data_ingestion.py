import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# Configure a basic logger
logging.basicConfig(
    level = logging.DEBUG,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)

# ----------------------------
# Load test_size parameter safely
# ----------------------------
def load_params(params_path):
    """
    Load test_size parameter from params.yaml.
    """
    try:
        logging.info(f"Loading parameters from {params_path}")
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        
        test_size = params.get("dataingestion", {}).get("test_size", None)
        
        if test_size is None:
            raise ValueError("'test_size' not found in params.yaml under 'dataingestion'")
            
        logging.info(f"Successfully loaded test_size = {test_size}")
        return test_size
    except FileNotFoundError:
        logging.error(f"Error: params.yaml file not found at {params_path}")
        return None
    except yaml.YAMLError as e:
        logging.error(f"YAML parsing error: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error while loading parameters: {e}")
        return None


# ----------------------------
# Load and combine datasets
# ----------------------------
def load_and_combine_news_datasets(fake_news_path, true_news_path):
    """
    Loads fake and true news datasets, assigns labels, and combines them.
    """
    try:
        logging.info(f"Checking file paths: fake='{fake_news_path}', true='{true_news_path}'")
        if not os.path.exists(fake_news_path):
            raise FileNotFoundError(f"Fake news file not found: {fake_news_path}")
        if not os.path.exists(true_news_path):
            raise FileNotFoundError(f"True news file not found: {true_news_path}")

        logging.info(f"Loading fake news data from {fake_news_path}")
        fake_news_df = pd.read_csv(fake_news_path)
        fake_news_df["label"] = "fake"
        logging.info(f"Loaded fake news: {fake_news_path} | Shape: {fake_news_df.shape}")

        logging.info(f"Loading true news data from {true_news_path}")
        true_news_df = pd.read_csv(true_news_path)
        true_news_df["label"] = "true"
        logging.info(f"Loaded true news: {true_news_path} | Shape: {true_news_df.shape}")

        logging.info("Combining datasets...")
        combined_df = pd.concat([true_news_df, fake_news_df], ignore_index=True)
        logging.info(f"Combined dataset created | Shape: {combined_df.shape}")
        
        return combined_df

    except pd.errors.EmptyDataError:
        logging.error("Error: One or both CSV files are empty.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error while loading datasets: {e}")
        return None


# ----------------------------
# Preprocess dataset
# ----------------------------
def preprocess_news_data(df):
    """
    Preprocesses the combined news dataset.
    """
    if df is None:
        logging.error("Input DataFrame is None. Aborting preprocessing.")
        return None

    try:
        logging.info("Starting data preprocessing...")
        df_processed = df.copy()

        # Convert date column to datetime
        if "date" in df_processed.columns:
            df_processed["date"] = pd.to_datetime(df_processed["date"], errors='coerce', dayfirst=True, format="mixed")
            dropped_rows = df_processed["date"].isna().sum()
            df_processed.dropna(subset=["date"], inplace=True)
            logging.info(f"Dropped {dropped_rows} rows due to invalid dates.")
        else:
            logging.warning("Warning: 'date' column missing. Skipping date processing.")

        # Replace labels
        if "label" in df_processed.columns:
            df_processed["label"].replace({"fake": 0, "true": 1}, inplace=True)
            logging.info("Labels 'fake' and 'true' replaced with 0 and 1.")
        else:
            logging.warning("Warning: 'label' column missing.")

        logging.info(f"Preprocessing complete. Final shape: {df_processed.shape}")
        return df_processed

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        return None


# ----------------------------
# Save train/test splits safely
# ----------------------------
def save_data(data_path, train_data, test_data):
    """
    Save train and test data to CSV files.
    """
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logging.info(f"Data saved in {data_path}")
    except PermissionError:
        logging.error(f"Permission denied: Could not write to {data_path}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")


# ----------------------------
# Main workflow
# ----------------------------
def main():
    params_path = "params.yaml"
    test_size = load_params(params_path)

    if test_size is None:
        logging.error("Aborting: test_size parameter could not be loaded.")
        return

    df = load_and_combine_news_datasets(
        fake_news_path=r"C:\Users\Varun\Downloads\Saurav\Outsource 360 internship\Project\Datasets\Fake-Real-News\Fake.csv",
        true_news_path=r"C:\Users\Varun\Downloads\Saurav\Outsource 360 internship\Project\Datasets\Fake-Real-News\True.csv"
    )

    final_df = preprocess_news_data(df)
    if final_df is None:
        logging.error("Aborting: Preprocessing failed.")
        return

    try:
        logging.info("Performing train/test split...")
        train_data, test_data = train_test_split(final_df, test_size=test_size, stratify=final_df["label"], random_state=42)
        logging.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    except Exception as e:
        logging.error(f"Error during train/test split: {e}")
        return

    data_path = os.path.join("data", "raw")
    save_data(data_path, train_data, test_data)


if __name__ == "__main__":
    main()