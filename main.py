""" The main entrypoint for the application. """
from kaggle.api.kaggle_api_extended import KaggleApi
import logging
import json
import os


# Set up logging
logging.basicConfig(filename='sun_predictor.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


from src.data.data_fetcher import get_raw_data
from src.visualization.plotting import plot_all_features

logger.info("Starting the application...")


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

    PATH_TO_RESULTS = "results/output/"
    csv_file_path  = PATH_TO_RESULTS + file_name + suffix
    # Initialize the Kaggle API
    username, key = get_kaggle_credentials()

    api = KaggleApi()
    api.authenticate()
    # print(f"competitions_list: {api.competitions_list()}")

    COMPETITION_NAME: str = "solar-energy-prediction-forecasting-competition"
    api.competition_download_files(COMPETITION_NAME)
    # Submit the CSV file
    logger.info(f"Submitting {csv_file_path} to {COMPETITION_NAME}, with message: {message}...")
    # Signature: competition_submit(file_name, message, competition,quiet=False)
    api.competition_submit(csv_file_path, message, COMPETITION_NAME)
    logger.info(f"File {csv_file_path} successfully submitted to {COMPETITION_NAME}!")

logger.info("Submitting the CSV file to Kaggle...")
submit_to_kaggle("submission", "Random Values, no ML, automatic deployment test")