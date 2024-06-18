from dataclasses import dataclass
from pathlib import Path
from typing import List

# Data Ingestion
@dataclass(frozen=True)
class DataIngestionConfig:
    # Kaggle Credentials from secrets.yaml
    username: str
    token: str
    # config.yaml
    download_dir: Path
    dataset_id: str
    
# Data Preparation
@dataclass(frozen=True)
class DataPreparationConfig:
    # config.yaml
    data_directory: Path
    metadata_csv: Path
    out_metadata_csv: Path
    # params.yaml
    random_state: int
    train_val_split: List[float]
    
# Training
@dataclass(frozen=True)
class TrainingConfig:
    # config.yaml
    metadata_csv: Path
    logs_dir: Path
    # params.yaml
    architecture: str
    encoder: str
    encoder_weights: str
    n_classes: int
    n_channels: int
    epochs: int
    lr: float
    batch_size: int
    device: str
    num_workers: int
    resize_dimension: int
    checkpoint_path: Path
    encoder: str
    optimizer: str
    loss: str
    apply_preprocessing: bool
    tune_lr: bool
    dev_run: bool
    
# Evaluation
@dataclass(frozen=True)
class EvalutationConfig:
    # config.yaml
    models_dir: Path
    results_dir: Path
    metrics_filepath: Path
    metadata_csv: Path
    # params.yaml
    batch_size: int
    num_workers: int
    resize_dimension: int
    device: str
    model_path: Path
    save_predictions: bool
    f1_threshold: float

from DeepGlobeRoadExtraction import CONFIG_FILE_PATH, SECRETS_FILE_PATH, PARAMS_FILE_PATH
from DeepGlobeRoadExtraction.utils.common import read_yaml, create_directories

class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, secrets_filepath = SECRETS_FILE_PATH, params_filepath = PARAMS_FILE_PATH) -> None:
        self.config = read_yaml(config_filepath)
        self.secrets = read_yaml(secrets_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.data_ingestion.download_dir])
    
    # Data Ingestion
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
    
    # Data Preparation
    def get_data_preparation_config(self) -> DataPreparationConfig:
        config = self.config.data_preparation
        params = self.params
        cfg = DataPreparationConfig(
            data_directory=Path(config.data_dir),
            metadata_csv=Path(config.metadata_csv),
            random_state=params.random_state,
            train_val_split=params.train_val_split,
            out_metadata_csv=Path(config.out_metadata_csv)
        )
        return cfg
    
    # Training
    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        params = self.params
        cfg = TrainingConfig(
            metadata_csv=Path(config.metadata_csv),
            logs_dir=Path(config.logs_dir),
            architecture=params.architecture,
            encoder=params.encoder,
            encoder_weights=params.encoder_weights,
            n_classes=params.n_classes,
            n_channels=params.n_channels,
            epochs=params.epochs,
            lr=params.lr,
            batch_size=params.batch_size,
            device=params.device,
            num_workers=params.num_workers,
            resize_dimension=params.resize_dimension,
            checkpoint_path=None if params.checkpoint_path == 'None' else Path(params.checkpoint_path),
            optimizer=params.optimizer,
            loss=params.loss,
            apply_preprocessing=params.apply_preprocessing,
            tune_lr=params.tune_lr,
            dev_run=params.dev_run
        )
        return cfg
    
    # Evaluation
    def get_evaluation_config(self) -> EvalutationConfig:
        config = self.config.evaluation
        params = self.params
        plots_dir = Path(config.results_dir) / "plots"
        create_directories([config.models_dir, config.results_dir, plots_dir])
        
        cfg = EvalutationConfig(
            models_dir=config.models_dir,
            results_dir=config.results_dir,
            metrics_filepath=config.metrics_filepath,
            metadata_csv=config.metadata_csv,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            resize_dimension=params.resize_dimension,
            device=params.device,
            model_path=params.model_path,
            save_predictions=params.save_predictions,
            f1_threshold=params.f1_threshold
        )
        return cfg