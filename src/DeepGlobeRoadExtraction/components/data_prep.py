from DeepGlobeRoadExtraction.configuration import DataPrepConfig
from DeepGlobeRoadExtraction import logger
import pandas as pd
import os
from sklearn.model_selection import train_test_split

class DataPrepComponents:
    def __init__(self, config: DataPrepConfig) -> None:
        self.config = config
    
    def load_metadata(self) -> pd.DataFrame:
        logger.info(f'---------- Loading Metadata: {self.config.metadata_path} ----------')
        try:
            metadata_df = pd.read_csv(self.config.metadata_path)
            metadata_df = metadata_df[metadata_df['split'] == 'train']
            metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']] # Select Columns
            metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(lambda img_path: os.path.join(self.config.data_dir, img_path)) # Update image paths
            metadata_df['mask_path'] = metadata_df['mask_path'].apply(lambda img_path: os.path.join(self.config.data_dir, img_path)) # Update mask paths
            self.metadata_df = metadata_df
        except Exception as e:
            logger.error(f'Failed to load Metadata: {self.config.metadata_path}')
            logger.exception(e)
            
    def split_dataset(self):
        logger.info(f'---------- Splitting Training Data into Training, Testing, and Validation Sets ----------')
        try:
            metadata_df = self.metadata_df.sample(frac=1).reset_index(drop=True) # Shuffle DataFrame
            train_df, valid_df = train_test_split(metadata_df, train_size=self.config.train_val_test_split_ratio[0], random_state=self.config.random_state) # Split into training and val + test (combined)
            valid_df, test_df = train_test_split(valid_df, train_size=self.config.train_val_test_split_ratio[1]/(self.config.train_val_test_split_ratio[1] + self.config.train_val_test_split_ratio[2]), random_state=self.config.random_state) # Split val + test combined into val and test sets
            train_df['group'] = 'train'
            test_df['group'] = 'test'
            valid_df['group'] = 'val'
            # Concatenate DataFrames
            self.metadata = pd.concat([train_df, test_df, valid_df]) # Processed Metadata
            # Export as CSV
            self.metadata.to_csv(self.config.processed_metadata_path, index=False)
            del train_df, test_df, valid_df, metadata_df 
        except Exception as e:
            logger.info("Failed to Split Training Data")
            logger.exception(e)