import logging
import json
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Set up logging
logger = logging.getLogger(__name__)
PATH_TO_RESULTS = "results/output/"

def submit_newest_to_kaggle(message: str) -> None:
    newest_submission = get_newest_submission(PATH_TO_RESULTS)
    if newest_submission is None:
        logger.error("No CSV files found in the results directory!")
        return
    logger.info(f"Newest submission: {newest_submission}")
    print(f"Newest submission: {newest_submission}")
    submit_to_kaggle(newest_submission, message)


def get_newest_submission(directory: str) -> str:
     # List all files in the directory
    all_files = os.listdir(directory)
    
    # Filter out files that are not CSVs
    csv_files = [f for f in all_files if f.endswith('.csv')]
    
    # If there are no CSV files, return None
    if not csv_files:
        return None
    
    # Sort the CSV files by modification time
    newest_csv = max(csv_files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    
    return newest_csv


def submit_to_kaggle(file_name: str, message: str, suffix: str = ".csv") -> None:
    """
    Submit a CSV file to a Kaggle competition.
    
    Parameters:
    - csv_file_path (str): The path to the CSV file you want to submit.
    - message (str):
    """
    # Make sure that credentials are available
    username, key = get_kaggle_credentials()
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    # Initialize the Kaggle API
    username, key = get_kaggle_credentials()

    api = KaggleApi()
    api.authenticate()

    # Get the newest submission
    PATH_TO_RESULTS = "results/output/"
    csv_file_path  = PATH_TO_RESULTS + file_name
    
    # Add the suffix if it's not already there
    if not csv_file_path.endswith(suffix):
        csv_file_path += suffix

    COMPETITION_NAME: str = "solar-energy-prediction-forecasting-competition"
    api.competition_download_files(COMPETITION_NAME)
    # Submit the CSV file
    logger.info(f"Submitting {csv_file_path} to {COMPETITION_NAME}, with message: {message}...")
    # Signature: competition_submit(file_name, message, competition,quiet=False)
    api.competition_submit(csv_file_path, message, COMPETITION_NAME)
    logger.info(f"File {csv_file_path} successfully submitted to {COMPETITION_NAME}!")

def get_kaggle_credentials():
    # Path to the kaggle.json file (default location)
    kaggle_json_path = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    
    # Check if the file exists
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError(f"{kaggle_json_path} not found!")
    
    # Read the file and extract credentials
    with open(kaggle_json_path, 'r') as file:
        credentials = json.load(file)
        username = credentials.get("username")
        key = credentials.get("key")
    logger.info(f"Successfully retrieved Kaggle credentials for user {username}!")
    return username, key

if __name__ == "__main__":
    # Submit the newest CSV file to Kaggle
    logger.info("Submitting the CSV file to Kaggle...")
    submit_newest_to_kaggle("Random Values, no ML, automatic deployment test")