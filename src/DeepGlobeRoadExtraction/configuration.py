from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    # Kaggle Credentials from secrets.yaml
    username: str
    token: str
    # config.yaml
    download_dir: Path
    dataset_id: str

from DeepGlobeRoadExtraction import CONFIG_FILE_PATH, SECRETS_FILE_PATH
from DeepGlobeRoadExtraction.utils.common import read_yaml, create_directories

class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, secrets_filepath = SECRETS_FILE_PATH) -> None:
        self.config = read_yaml(config_filepath)
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