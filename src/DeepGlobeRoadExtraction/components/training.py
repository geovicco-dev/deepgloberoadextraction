from DeepGlobeRoadExtraction.configuration import TrainingConfig
from DeepGlobeRoadExtraction import logger
from DeepGlobeRoadExtraction.utils.dataloader import RoadsDataModule, get_training_augmentation, get_preprocessing, get_preprocessing_function
from DeepGlobeRoadExtraction.utils.model import SegmentationModel
import warnings; warnings.filterwarnings("ignore")
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os
import numpy as np
import random
torch.set_float32_matmul_precision('medium')

class TrainingComponents:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        
    def create_dataloaders(self):
        logger.info(f'------------- Creating Dataloaders -------------')
        if self.config.apply_preprocessing:
            logger.info('------------->>> Applying Preprocessing <<<-------------')
            self.dm = RoadsDataModule(
                metadata_csv=self.config.metadata_csv,
                augmentation=get_training_augmentation(),
                preprocessing=get_preprocessing(get_preprocessing_function(self.config.encoder, self.config.encoder_weights)),
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                resize_dimensions=self.config.resize_dimension
            )
        else:
            logger.info('------------->>> Skipping Applying Preprocessing <<<-------------')
            self.dm = RoadsDataModule(
                metadata_csv=self.config.metadata_csv,
                augmentation=get_training_augmentation(),
                preprocessing=None,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                resize_dimensions=self.config.resize_dimension
            )
            
        # Plot sample from the training dataset
    @staticmethod
    def plot_train_batch(dm, n_samples=4, randomised=True):
        dm.setup('fit')
        # Get the train dataloader
        dataloader = dm.train_dataloader()

        if randomised:
            # Randomly select a batch of data
            x, y = random.choice(list(dataloader))
        else:
            # Select from first batch of data
            x, y = next(iter(dataloader))

        # Plot the results
        fig, axs = plt.subplots(n_samples, 2, figsize=(10, n_samples*5))
        for i in range(n_samples):
            # Plot the image
            image = x[i].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            # Get Vmin and Vmax as 2nd and 98th percentile
            vmin = np.percentile(image, 2)
            vmax = np.percentile(image, 98)
            axs[i, 0].imshow(image, vmin=vmin, vmax=vmax)
            axs[i, 0].axis('off')
            if i == 0:
                axs[i, 0].set_title('Image')
            
            # Plot the ground truth mask
            ground_truth_mask = y[i].cpu().numpy().squeeze()  # (1, H, W) -> (H, W)
            axs[i, 1].imshow(ground_truth_mask, cmap='binary_r')
            axs[i, 1].axis('off')
            if i == 0:
                axs[i, 1].set_title('Ground Truth Mask')

        plt.tight_layout()
        plt.show()
    
    def initialise_model(self):
        logger.info(f'------------- Inistialising Model: Architecture: {self.config.architecture} | Encoder: {self.config.encoder} | Encoder Weights: {self.config.encoder_weights} -------------')
        self.model = SegmentationModel(
            architecture=self.config.architecture,
            n_channels=self.config.n_channels,
            n_classes=self.config.n_classes,
            lr=self.config.lr,
            encoder=self.config.encoder,
            encoder_weights=self.config.encoder_weights,
            loss=self.config.loss,
            optimizer=self.config.optimizer,
        )
    
    def create_callbacks(self):
        logger.info('------------- Creating Callbacks -------------')
        ### Define Checkpoints for Early Stopping, Tensorboard Summary Writer, and Best Checkpoint Saving
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_lightning.loggers import TensorBoardLogger

        # Early stopping callback
        self.early_stopping = EarlyStopping(
            monitor='val_loss',  # Metric to monitor
            patience=10,          # Number of epochs with no improvement after which training will be stopped
            verbose=True,
            mode='min'           # Mode can be 'min' for minimizing the monitored metric or 'max' for maximizing it
        )

        # Model checkpoint callback
        self.checkpoint_callback = ModelCheckpoint(
            monitor='val_f1',   # Metric to monitor
            filename='{epoch:02d}-{val_f1:.2f}',  # Filename format
            save_top_k=1,         # Save the top k models
            mode='max',           # Mode can be 'min' or 'max'
            verbose=True,
            dirpath=self.config.logs_dir.joinpath(f'{self.config.architecture}_{self.config.encoder}/checkpoints')
        )

        # TensorBoard logger
        self.tensorboard_logger = TensorBoardLogger(
            save_dir=self.config.logs_dir,     # Directory to save the logs
            name=f"{self.config.architecture}_{self.config.encoder}"       # Experiment name
        )

        from pytorch_lightning.callbacks import LearningRateMonitor
        # Learning rate monitor
        self.lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
    def tune_lr(self):
        if self.config.tune_lr:
            from pytorch_lightning.tuner.tuning import Tuner
            logger.info('------------- Tunning Learning Rate -------------')
            # Define a separate trainer for hyperparameter tuning
            self.tuning_trainer = pl.Trainer(
                accelerator=self.config.device,
                precision="16-mixed",
                logger=self.tensorboard_logger,
                callbacks=None,
                max_epochs=5  # Set this to a low number for faster tuning
            )

            self.dm.setup('fit')

            # Hyperparameter tuning
            self.tuner = Tuner(self.tuning_trainer)
            self.new_lr = self.tuner.lr_find(self.model, train_dataloaders=self.dm.train_dataloader(), val_dataloaders=self.dm.val_dataloader()).suggestion()
            logger.info(f'Suggested learning rate: {self.new_lr}')
        else:
            logger.info('------------- Skipping Tunning Learning Rate -------------')
            
        
    def create_trainer(self):
        logger.info(f'------------- Training Model: {self.config.architecture} with {self.config.encoder} Encoder -------------')
        self.trainer = pl.Trainer(
            accelerator=self.config.device,
            max_epochs=self.config.epochs,
            precision="16-mixed",
            logger= self.tensorboard_logger if hasattr(self, 'tensorboard_logger') else None,
            callbacks=[self.early_stopping, self.checkpoint_callback, self.lr_monitor],
            enable_progress_bar=True,
            fast_dev_run=self.config.dev_run,
        )
    
    def train(self):
        logger.info('------------- Training Started -------------')
        self.dm.setup('fit')
        if self.config.checkpoint_path is not None and os.path.exists(self.config.checkpoint_path):
            logger.info(f'Resuming training from checkpoint: {self.config.checkpoint_path}')
            self.trainer.fit(model=self.model, train_dataloaders=self.dm.train_dataloader(), val_dataloaders=self.dm.val_dataloader(), ckpt_path=self.config.checkpoint_path)
        else:
            self.trainer.fit(model=self.model, train_dataloaders=self.dm.train_dataloader(), val_dataloaders=self.dm.val_dataloader())
        logger.info('------------- Training Completed -------------')