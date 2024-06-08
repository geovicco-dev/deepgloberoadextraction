# https://medium.com/@r1j1nghimire/semantic-segmentation-using-deeplabv3-from-scratch-b1ff57a27be
# https://www.cnblogs.com/vincent1997/p/10889430.html?source=post_page-----b1ff57a27be--------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet_50(nn.Module):
    def __init__(self, block, layers, input_channels=3, output_layer=None):
        super(ResNet_50, self).__init__()
        self.in_planes = 64
        self.output_layer = output_layer

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.output_layer == 'layer1':
            return x
        x = self.layer2(x)
        if self.output_layer == 'layer2':
            return x
        x = self.layer3(x)
        if self.output_layer == 'layer3':
            return x
        x = self.layer4(x)
        if self.output_layer == 'layer4':
            return x
        return x

class Atrous_Convolution(nn.Module):
    def __init__(self, input_channels, kernel_size, pad, dilation_rate, output_channels=256):
        super(Atrous_Convolution, self).__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=pad, dilation=dilation_rate, bias=False)
        self.batchnorm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class ASSP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASSP, self).__init__()
        self.conv_1x1 = Atrous_Convolution(input_channels=in_channels, output_channels=out_channels, kernel_size=1, pad=0, dilation_rate=1)
        self.conv_6x6 = Atrous_Convolution(input_channels=in_channels, output_channels=out_channels, kernel_size=3, pad=6, dilation_rate=6)
        self.conv_12x12 = Atrous_Convolution(input_channels=in_channels, output_channels=out_channels, kernel_size=3, pad=12, dilation_rate=12)
        self.conv_18x18 = Atrous_Convolution(input_channels=in_channels, output_channels=out_channels, kernel_size=3, pad=18, dilation_rate=18)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        self.final_conv = Atrous_Convolution(input_channels=out_channels * 5, output_channels=out_channels, kernel_size=1, pad=0, dilation_rate=1)

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_6x6 = self.conv_6x6(x)
        x_12x12 = self.conv_12x12(x)
        x_18x18 = self.conv_18x18(x)
        img_pool_opt = self.image_pool(x)
        img_pool_opt = F.interpolate(img_pool_opt, size=x_18x18.size()[2:], mode='bilinear', align_corners=True)
        concat = torch.cat((x_1x1, x_6x6, x_12x12, x_18x18, img_pool_opt), dim=1)
        x_final_conv = self.final_conv(concat)
        return x_final_conv

class Deeplabv3Plus(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super(Deeplabv3Plus, self).__init__()

        self.backbone = ResNet_50(Bottleneck, [3, 4, 6, 3], input_channels=input_channels, output_layer='layer3')
        self.low_level_features = ResNet_50(Bottleneck, [3, 4, 6, 3], input_channels=input_channels, output_layer='layer1')
        self.assp = ASSP(in_channels=1024, out_channels=256)
        self.conv1x1 = Atrous_Convolution(input_channels=256, output_channels=48, kernel_size=1, dilation_rate=1, pad=0)
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x_backbone = self.backbone(x)
        x_low_level = self.low_level_features(x)
        x_assp = self.assp(x_backbone)
        x_assp_upsampled = F.interpolate(x_assp, scale_factor=4, mode='bilinear', align_corners=True)
        x_conv1x1 = self.conv1x1(x_low_level)
        x_cat = torch.cat([x_conv1x1, x_assp_upsampled], dim=1)
        x_3x3 = self.conv_3x3(x_cat)
        x_3x3_upscaled = F.interpolate(x_3x3, scale_factor=4, mode='bilinear', align_corners=True)
        x_out = self.classifier(x_3x3_upscaled)
        return x_out
    
# Loss Function
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return bce_loss + dice_loss

import pytorch_lightning as pl
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
    
# Wrap it in a PyTorch Lightning Module
class DeepLabV3PlusModel(pl.LightningModule):
    def __init__(self, n_channels, n_classes, lr):
        super().__init__()
        # self.save_hyperparameters() # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.lr = lr
        
        # Initialize model
        self.model = Deeplabv3Plus(num_classes=self.n_classes, input_channels=self.n_channels)
        
        # Initialize metrics
        self.precision = torchmetrics.Precision(task="binary", threshold=0.5)
        self.recall = torchmetrics.Recall(task="binary", threshold=0.5)
        self.f1 = torchmetrics.F1Score(task="binary", threshold=0.5)
        self.jaccard = torchmetrics.JaccardIndex(task="binary", threshold=0.5)
        
        # Define Loss
        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = CombinedLoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch  # x is the input image and y is the corresponding segmentation mask
        y_hat = self(x)  # Forward pass
        loss = self.loss_fn(y_hat, y)  # Compute loss
        y_hat_argmax = (torch.sigmoid(y_hat) > 0.5).float()  # Apply sigmoid and threshold at 0.5

        # Torch metric
        tm_rec = self.recall(y_hat_argmax, y)
        tm_prec = self.precision(y_hat_argmax, y)
        tm_iou = self.jaccard(y_hat_argmax, y)
        tm_f1 = self.f1(y_hat_argmax, y)

        self.log_dict({
            "train_loss": loss,
            "train_iou": tm_iou,
            "train_f1": tm_f1,
            "train_precision": tm_prec,
            "train_recall": tm_rec,
            "train_lr": self.lr,
        }, prog_bar=True, on_step=False, on_epoch=True)
        
        # log values
        if self.logger:
            self.logger.experiment.add_scalar('Train/F1Score', tm_f1, self.global_step)
            self.logger.experiment.add_scalar('Train/IoU', tm_iou, self.global_step)
            self.logger.experiment.add_scalar('Train/Recall', tm_rec, self.global_step)
            self.logger.experiment.add_scalar('Train/Precision', tm_prec, self.global_step)
            self.logger.experiment.add_scalar('Train/BCEwithLogits', loss, self.global_step)
            self.logger.experiment.add_scalar('Train/LearningRate', self.lr, self.global_step)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)  # Compute loss
        y_hat_argmax = (torch.sigmoid(y_hat) > 0.5).float()  # Apply sigmoid and threshold at 0.5

        # Torch metric
        tm_rec = self.recall(y_hat_argmax,y)
        tm_prec = self.precision(y_hat_argmax,y)
        tm_iou = self.jaccard(y_hat_argmax,y)
        tm_f1 = self.f1(y_hat_argmax,y)
        
        log_dict = {
            "val_loss":loss,
            "val_iou":tm_iou,
            "val_f1":tm_f1,
            "val_precision":tm_prec,
            "val_recall":tm_rec,
        }
        
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)
        
        if self.logger:
            self.logger.experiment.add_scalar('Valid/F1Score', tm_f1, self.global_step)
            self.logger.experiment.add_scalar('Valid/IoU', tm_iou, self.global_step)
            self.logger.experiment.add_scalar('Valid/Recall', tm_rec, self.global_step)
            self.logger.experiment.add_scalar('Valid/Precision', tm_prec, self.global_step)
            self.logger.experiment.add_scalar('Valid/BCEwithLogits', loss, self.global_step)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        y_hat_argmax = (torch.sigmoid(y_hat) > 0.5).float()  # Apply sigmoid and threshold at 0.5

        # Torch metric
        tm_rec = self.recall(y_hat_argmax,y)
        tm_prec = self.precision(y_hat_argmax,y)
        tm_iou = self.jaccard(y_hat_argmax,y)
        tm_f1 = self.f1(y_hat_argmax,y)
        
        log_dict = {
            "test_loss":loss,
            "test_iou":tm_iou,
            "test_f1":tm_f1,
            "test_precision":tm_prec,
            "test_recall":tm_rec,
        }
        
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True),
            'monitor': 'val_loss'
        }
        return [optimizer], [lr_scheduler]