from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    # Kaggle Credentials from secrets.yaml
    username: str
    token: str
    # config.yaml
    download_dir: Path
    dataset_id: str

@dataclass(frozen=True)
class DataPrepConfig:
    # params.yaml
    random_state: int
    train_val_test_split_ratio: List[float]
    # config.yaml
    data_dir: Path
    metadata_path: Path
    processed_metadata_path: Path

from DeepGlobeRoadExtraction import CONFIG_FILE_PATH, SECRETS_FILE_PATH, PARAMS_FILE_PATH
from DeepGlobeRoadExtraction.utils.common import read_yaml, create_directories

class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, secrets_filepath = SECRETS_FILE_PATH, params_filepath = PARAMS_FILE_PATH) -> None:
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.secrets = read_yaml(secrets_filepath)
        create_directories([self.config.data_ingestion.download_dir])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        secrets = self.secrets.kaggle
        cfg = DataIngestionConfig(
            download_dir=Path(config.download_dir),
            dataset_id=config.dataset_id,
            username=secrets.username,
            token=secrets.token
        )
        return cfg
    
    def get_data_prep_config(self) -> DataPrepConfig:
        config = self.config.data_preparation
        params = self.params
        cfg = DataPrepConfig(
            random_state=params.random_state,
            train_val_test_split_ratio=params.train_val_test_split_ratio,
            data_dir=config.data_dir,
            metadata_path=config.metadata_path,
            processed_metadata_path=config.processed_metadata_path
        )
        return cfg