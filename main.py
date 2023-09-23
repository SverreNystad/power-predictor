""" The main entrypoint for the application. """
import logging


# Set up logging
logging.basicConfig(filename='sun_predictor.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


from src.data.data_fetcher import get_raw_data
from src.visualization.plotting import plot_all_features

logger.info("Starting the application...")
