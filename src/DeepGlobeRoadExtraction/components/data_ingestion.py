from DeepGlobeRoadExtraction.configuration import DataIngestionConfig
import os
import subprocess
import json
from DeepGlobeRoadExtraction import logger

class DataIngestionComponents:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def initialise_kaggle(self):
        logger.info(f'---------- Initialising Kaggle Account ----------')
        # Set Path for Kaggle Configration File
        KAGGLE_CONFIG_DIR = os.path.join(os.path.expandvars('$HOME'), '.kaggle')
        KAGGLE_CONFIG_FILE = os.path.join(KAGGLE_CONFIG_DIR, 'kaggle.json')
        
        # Check if kaggle.json already exists and is not empty
        if os.path.exists(KAGGLE_CONFIG_FILE) and os.path.getsize(KAGGLE_CONFIG_FILE) > 0:
            logger.warning(f'---> Kaggle Account Credentials Found ==> {KAGGLE_CONFIG_FILE}. Remove this file and re-initialse if API token is invalid or has expired.')
            return
        
        # Otherwise create .kaggle directory
        os.makedirs(KAGGLE_CONFIG_DIR, exist_ok=True)
        
        try:
            username = self.config.username
            token = self.config.token
            api_dict = {"username": username, "key": token}
            
            # Create a kaggle.json file inside .kaggle folder and add your credentials
            with open(KAGGLE_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(api_dict, f)
            
            # Change File Permissions
            cmd = f"chmod 600 {KAGGLE_CONFIG_FILE}"
            output = subprocess.check_output(cmd.split(" "))
            output = output.decode(encoding="utf-8")
        except Exception as e:
            logger.error('Failed to Initialise Kaggle Account!')
            logger.exception(e)
            raise e
        
    # Download Kaggle Dataset
    def download_dataset(self):
        from kaggle.api.kaggle_api_extended import KaggleApi
        logger.info(f'---------- Downloading Kaggle Dataset: {self.config.dataset_id} ----------')
        try:
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                dataset=self.config.dataset_id,
                path=self.config.download_dir,
                unzip=True,
                force=False,
                quiet=True
            )
            logger.info('---> Download Complete!')
        except Exception as e:
            logger.error('Kaggle dataset download failed!')
            logger.exception(e)
            raise e