import pytorch_lightning as pl
import segmentation_models_pytorch as sm_torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.optim as optim

def get_optimizer(optimizer_name, parameters, lr):
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(parameters, lr=lr)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(parameters, lr=lr)
    elif optimizer_name == 'adadelta':
        optimizer = optim.Adadelta(parameters, lr=lr)
    elif optimizer_name == 'adamax':
        optimizer = optim.Adamax(parameters, lr=lr)
    elif optimizer_name == 'asgd':
        optimizer = optim.ASGD(parameters, lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(parameters, lr=lr)
    elif optimizer_name == 'adagrad':
        optimizer = optim.Adagrad(parameters, lr=lr)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=lr)
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')
    
    return optimizer

class SegmentationModel(pl.LightningModule):
    def __init__(self, architecture, n_channels, n_classes, lr, encoder, encoder_weights, loss, optimizer):
        super().__init__()
        self.save_hyperparameters() # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.lr = lr
        self.architecture = architecture
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.loss = loss
        self.mode = 'binary' if self.n_classes == 1 else 'multiclass'
        self.optimizer = optimizer
        
        self.model_dict = {
            'Unet': sm_torch.Unet,
            'DeepLabV3Plus': sm_torch.DeepLabV3Plus,
            'DeepLabV3': sm_torch.DeepLabV3,
            'UnetPlusPlus': sm_torch.UnetPlusPlus,
            'Linknet': sm_torch.Linknet,
            'PSPNet': sm_torch.PSPNet,
            'FPN': sm_torch.FPN,
            'MAnet': sm_torch.MAnet,
            'PAN': sm_torch.PAN
        }
        
        self.loss_dict = {
            'DiceLoss': sm_torch.losses.DiceLoss,
            'FocalLoss': sm_torch.losses.FocalLoss,
            'TverskyLoss': sm_torch.losses.TverskyLoss,
            'JaccardLoss': sm_torch.losses.JaccardLoss,
            'LovaszLoss': sm_torch.losses.LovaszLoss,
            'SoftBCEWithLogitsLoss': sm_torch.losses.SoftBCEWithLogitsLoss,
            'SoftCrossEntropyLoss': sm_torch.losses.SoftCrossEntropyLoss,
            'MCCLoss': sm_torch.losses.MCCLoss
            
        }
        
        # Initialize model
        self.model = self.model_dict[self.architecture](encoder_name=self.encoder, encoder_weights=self.encoder_weights, in_channels=self.n_channels, classes=self.n_classes)
        
        # Loss function
        if self.loss in ['DiceLoss', 'FocalLoss', 'TverskyLoss', 'JaccardLoss', 'LovaszLoss']:
            self.loss_fn = self.loss_dict[self.loss](mode=self.mode)
        else:
            self.loss_fn = self.loss_dict[self.loss]()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch  # x is the input image and y is the corresponding segmentation mask
        y_hat = self(x)  # Forward pass
        loss = self.loss_fn(y_hat, y)  # Compute loss
        
        tp, fp, fn, tn = sm_torch.metrics.get_stats(y_hat, y, mode=self.mode, threshold=0.5)
        
        iou_score = sm_torch.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = sm_torch.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        f2_score = sm_torch.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        recall = sm_torch.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        precision = sm_torch.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")

        self.log_dict({
            "train_loss": loss,
            "train_iou": iou_score,
            "train_f1": f1_score,
            "train_f2": f2_score,
            "train_precision": precision,
            "train_recall": recall,
        }, prog_bar=True, on_step=False, on_epoch=True)
        
        # log values
        if self.logger:
            self.logger.experiment.add_scalar('Train/F1Score', f1_score, self.global_step)
            self.logger.experiment.add_scalar('Train/F2Score', f2_score, self.global_step)
            self.logger.experiment.add_scalar('Train/IoU', iou_score, self.global_step)
            self.logger.experiment.add_scalar('Train/Recall', recall, self.global_step)
            self.logger.experiment.add_scalar('Train/Precision', precision, self.global_step)
            self.logger.experiment.add_scalar('Train/Loss', loss, self.global_step)
            self.logger.experiment.add_scalar('Train/LearningRate', self.lr, self.global_step)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)  # Compute loss
        
        tp, fp, fn, tn = sm_torch.metrics.get_stats(y_hat, y, mode=self.mode, threshold=0.5)
        
        iou_score = sm_torch.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = sm_torch.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        f2_score = sm_torch.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        recall = sm_torch.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        precision = sm_torch.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")
        
        self.log_dict({
            "val_loss": loss,
            "val_iou": iou_score,
            "val_f1": f1_score,
            "val_f2": f2_score,
            "val_precision": precision,
            "val_recall": recall,
        }, prog_bar=True, on_step=False, on_epoch=True)
        
        # log values
        if self.logger:
            self.logger.experiment.add_scalar('Validation/F1Score', f1_score, self.global_step)
            self.logger.experiment.add_scalar('Validation/F2Score', f2_score, self.global_step)
            self.logger.experiment.add_scalar('Validation/IoU', iou_score, self.global_step)
            self.logger.experiment.add_scalar('Validation/Recall', recall, self.global_step)
            self.logger.experiment.add_scalar('Validation/Precision', precision, self.global_step)
            self.logger.experiment.add_scalar('Validation/Loss', loss, self.global_step)
            self.logger.experiment.add_scalar('Validation/LearningRate', self.lr, self.global_step)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        tp, fp, fn, tn = sm_torch.metrics.get_stats(y_hat, y, mode=self.mode, threshold=0.5)
        
        iou_score = sm_torch.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = sm_torch.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        f2_score = sm_torch.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        recall = sm_torch.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        precision = sm_torch.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")
        
        log_dict = {
            "test_loss": loss,
            "test_iou": iou_score,
            "test_f1": f1_score,
            "test_f2": f2_score,
            "test_precision": precision,
            "test_recall": recall,
        }

        # log values
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)
        
        return log_dict
    
    def on_validation_epoch_end(self):
        metrics = self.trainer.logged_metrics

        # Ensure metrics are iterable before attempting to stack them
        mean_outputs = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                mean_outputs[k] = v  # Use the scalar value directly
            elif isinstance(v, list) and all(isinstance(i, torch.Tensor) for i in v):
                mean_outputs[k] = torch.stack(v).mean()  # Calculate the mean if it's a list of tensors
            else:
                mean_outputs[k] = torch.tensor(v).mean()  # Default case, convert to tensor and calculate mean

        # Log the mean metrics
        self.log_dict(mean_outputs, prog_bar=True)
        
    # def configure_optimizers(self):
    #     opt = torch.optim.Adam(self.parameters(), lr=self.lr)
    #     sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    #     return [opt], [sch]
    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.optimizer, self.parameters(), self.lr)
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5),
            'monitor': 'val_loss'
        }
        return [optimizer], [lr_scheduler]