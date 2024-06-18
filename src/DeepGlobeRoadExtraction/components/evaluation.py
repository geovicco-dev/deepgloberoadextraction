from DeepGlobeRoadExtraction.configuration import EvalutationConfig
from DeepGlobeRoadExtraction import logger
from DeepGlobeRoadExtraction.utils.dataloader import RoadsDataModule
from DeepGlobeRoadExtraction.utils.model import SegmentationModel
from DeepGlobeRoadExtraction.utils.common import save_json
import warnings; warnings.filterwarnings("ignore")
import torch
from pathlib import Path
import os
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
torch.set_float32_matmul_precision('medium')

class EvaluationComponents:
    def __init__(self, config: EvalutationConfig):
        self.config = config
        
    def create_dataloaders(self):
        self.dm = RoadsDataModule(
                metadata_csv=self.config.metadata_csv,
                augmentation=None,
                preprocessing=None,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                resize_dimensions=self.config.resize_dimension
        )
        
    def load_model(self):
        if self.config.model_path is not None and os.path.exists(self.config.model_path):
            logger.info('------------- Loading Checkpoint -------------')
            logger.info(f'Loading checkpoint from {self.config.model_path}')
            try:
                self.model = SegmentationModel.load_from_checkpoint(self.config.model_path)
                logger.info('Checkpoint loaded successfully')
            except Exception as e:
                logger.error(f'Failed to load checkpoint: {e}')
    
    def create_trainer(self):
        logger.info(f'------------- Creating Trainer -------------')
        self.trainer = pl.Trainer(
            accelerator=self.config.device,
            # max_epochs=self.config.epochs,
            inference_mode=True,
            precision="16-mixed",
            logger=self.tensorboard_logger if hasattr(self, 'tensorboard_logger') else None,
            callbacks=None,
            enable_progress_bar=True,
        )
                
    def evaluate(self):
        logger.info('------------- Evaluating Model -------------')
        self.dm.setup('test')
        try:
            test_results = self.trainer.test(model=self.model, dataloaders=self.dm.test_dataloader())
            
            # Extract relevant metrics from test_results
            # Assuming test_results is a list of dictionaries and contains y_true and y_pred
            self.metrics = test_results[0] if test_results else {}

            # Save metrics to a file
            logger.info(f'Saving metrics to: {self.config.metrics_filepath}')
            save_json(Path(self.config.metrics_filepath), self.metrics)  # Save the metrics to a file
            
            # Save Model as ONNX
            if self.metrics['test_f1'] > self.config.f1_threshold:
                logger.info(f"Test F1-Score ({self.metrics['test_f1']:.3f}) above threshold ({self.config.f1_threshold}), saving model as ONNX...")
                save_path = Path(self.config.models_dir).joinpath(f"{Path(self.config.model_path).parent.parent.name}_test_f1_{self.metrics['test_f1']:.3f}.onnx")
                logger.info(f'Saving model to: {save_path}')
                test_dataloader = self.dm.test_dataloader()
                input_sample, _ = next(iter(test_dataloader))
                input_sample = input_sample[0].unsqueeze(0)
                self.model.to_onnx(save_path, input_sample=input_sample, export_params=True)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            
    def save_predictions(self):
        if self.config.save_predictions:
            logger.info('------------- Saving Predictions -------------')
            # Get the test dataloader
            test_dataloader = self.dm.test_dataloader()

            # Randomly select a batch of data
            x, y = next(iter(test_dataloader))

            # Put the model in evaluation mode
            self.model.eval()

            # Disable gradients for this step
            with torch.no_grad():
                # Pass the data through the model
                y_hat = self.model(x)

            # Plot the results
            _, axs = plt.subplots(4, 4, figsize=(20, 20))  # Increase the number of columns to 3
            for i in range(4):
                # Plot the image
                image = np.transpose(x[i][:3, :, :]).squeeze()
                axs[i, 0].imshow(image)
                axs[i, 0].axis('off')
                if i == 0:
                    axs[i, 0].set_title('Image')
                
                # Plot the ground truth mask
                ground_truth_mask = np.transpose(y[i]).squeeze()
                axs[i, 1].imshow(ground_truth_mask, cmap='binary_r')
                axs[i, 1].axis('off')
                if i == 0:
                    axs[i, 1].set_title('Ground Truth Mask')

                # Plot the predicted mask - with and without thresholding
                predicted_mask = torch.sigmoid(np.transpose(y_hat[i])).squeeze()
                axs[i, 2].imshow(predicted_mask, cmap='RdYlGn')
                axs[i, 2].axis('off')
                if i == 0:
                    axs[i, 2].set_title('Predicted Mask (Raw)')

                axs[i, 3].imshow(predicted_mask > 0.5, cmap='binary_r')
                axs[i, 3].axis('off')
                if i == 0:
                    axs[i, 3].set_title('Predicted Mask (Thresholded)')
                
            # Remove empty subplots
            for j in range(4, 4):
                for i in range(4):
                    axs[i, j].axis('off')

            plt.tight_layout()
            # Save as PNG
            save_path = Path(self.config.results_dir).joinpath(f"plots/{Path(self.config.model_path).parent.parent.name}_test_f1_{self.metrics['test_f1']:.3f}.png")
            logger.info(f'Saving predictions to: {save_path}')
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            plt.show()