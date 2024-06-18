from DeepGlobeRoadExtraction.configuration import DataPreparationConfig
from DeepGlobeRoadExtraction import logger
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreparationComponents:
    def __init__(self, config: DataPreparationConfig) -> None:
        self.config = config
        
    def load_metadata(self):
        logger.info(f'------------- Loading Metadata -------------')
        metadata_df = pd.read_csv(self.config.metadata_csv) # Read Metadata
        metadata_df = metadata_df[metadata_df['split']=='train'] # Filter all rows that have 'train' in the 'split' column
        metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']] # Keep only 'image_id', 'sat_image_path' and 'mask_path' columns
        metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(lambda img_pth: os.path.join(self.config.data_directory, img_pth)) # Add data_directory to sat_image_path
        metadata_df['mask_path'] = metadata_df['mask_path'].apply(lambda img_pth: os.path.join(self.config.data_directory, img_pth)) # Add data_directory to mask_path
        self.metadata = metadata_df 
    
    def split_dataset(self):
        logger.info(f'------------- Splitting Training Dataset into Train and Validation -------------')
        metadata_df = self.metadata
        # Shuffle DataFrame
        metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
        # Perform split for train / val
        train_df, valid_df = train_test_split(metadata_df, train_size=self.config.train_val_split[0], random_state=self.config.random_state)
        valid_df, test_df = train_test_split(valid_df, train_size=self.config.train_val_split[1]/(self.config.train_val_split[1]+self.config.train_val_split[2]), random_state=self.config.random_state)
        train_df['group'] = 'train'
        valid_df['group'] = 'val'
        test_df['group'] = 'test'
        # Concatenate DataFrames
        self.metadata = pd.concat([train_df, valid_df, test_df])
        # Export Metadata
        self.metadata.to_csv(self.config.out_metadata_csv, index=False)
        del train_df, valid_df, test_df, metadata_df