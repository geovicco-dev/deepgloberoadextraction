import random
import rioxarray as rxr
import numpy as np
import pandas as pd
from pathlib import Path
import albumentations as A
import cv2
import torch
import lightning as L
from torch.utils.data import DataLoader
import warnings; warnings.filterwarnings('ignore')

#### PyTorch DataLoader ####
class RoadsDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        df: pd.DataFrame,
        split: str = 'train', 
        augmentation = None,
        resize_dimensions: int = 256 # H, W of Transformed Images
    ):
        self.image_paths = [fp for fp in df.loc[df['group'] == split]['sat_image_path'].tolist()]
        self.mask_paths = [fp for fp in df.loc[df['group'] == split]['mask_path'].tolist()]
        self.augmentation = augmentation
        self.resize_dimensions = resize_dimensions

    def __len__(self):
        # return length of
        return len(self.image_paths)
    
    @staticmethod
    def normalise_band(band):
        return (band - band.min()) / (band.max() - band.min())

    def __getitem__(self, i):
        # Process Image
        x_da = rxr.open_rasterio(self.image_paths[i])
        x_da = x_da.transpose('y', 'x', 'band')
        
        # Normalise Image
        x_da = x_da.groupby(group='band', squeeze=True).apply(self.normalise_band)
        
        image = x_da.data
        
        mask = rxr.open_rasterio(self.mask_paths[i]).sel(band=1).squeeze()
        mask = mask.transpose('y', 'x')
        mask = np.where(mask==255, 1, mask)  # Map 255 to 1
        mask = np.expand_dims(mask, axis=-1)
        mask = torch.from_numpy(mask.astype('long'))  # Convert to torch tensor
        
        # Replace nan values with 0
        image = np.nan_to_num(x_da).astype('float32')
        mask = np.nan_to_num(mask).astype('float32')
        
        # Define the target size
        target_size = (self.resize_dimensions, self.resize_dimensions)

        # Create a Resize transform
        resize_transform = A.Resize(*target_size, interpolation=cv2.INTER_LINEAR)

        # Ensure image and mask are numpy arrays
        image = np.array(image)
        mask = np.array(mask)

        # Apply the Resize transform to the image and mask
        transformed = resize_transform(image=image, mask=mask)
        image, mask = transformed['image'], transformed['mask']

        image = torch.from_numpy(image.copy())  # Make a copy of the numpy array
        mask = torch.from_numpy(mask.copy())  # Make a copy of the numpy array

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image.cpu().numpy(), mask=mask.cpu().numpy())
            image, mask = sample['image'], sample['mask']
            # Convert to torch tensors
            image = torch.from_numpy(image.copy())
            mask = torch.from_numpy(mask.copy())
     
        # Convert image and mask to (C, H, W) format
        image = image.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)
        
        return image, mask

#### PyTorch Lightning Data Module ####
class RoadsDataModule(L.LightningDataModule):
    def __init__(self, metadata_df, train_augmentation=None, val_augmentation=None, batch_size=8, prefetch_factor=8, num_workers=4, resize_dimensions=256):
        super().__init__()
        self.metadata = metadata_df
        self.train_augmentation = train_augmentation
        self.val_augmentation = val_augmentation
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.resize_dimensions = resize_dimensions

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = RoadsDataset(df=self.metadata, split='train', augmentation=self.train_augmentation, resize_dimensions=self.resize_dimensions)
            self.val_dataset = RoadsDataset(df=self.metadata, split='val', augmentation=self.val_augmentation, resize_dimensions=self.resize_dimensions)
        if stage == 'test':
            self.test_dataset = RoadsDataset(df=self.metadata, split='test', resize_dimensions=self.resize_dimensions)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, prefetch_factor=self.prefetch_factor, drop_last=True, pin_memory=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)
    
    
##### Data Augmentation #####
def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),  # horizontal flip
        A.VerticalFlip(p=0.5),  # vertical flip
        A.Rotate(limit=90),  # 90 degree rotation
        A.ColorJitter(p=0.2),  # color jitter
        A.RandomBrightnessContrast(p=0.2),  # random brightness and contrast
        A.RandomGamma(p=0.2),  # random gamma
        A.GaussNoise(p=0.2),  # gaussian noise
    ]
    return A.Compose(train_transform, is_check_shapes=False)
