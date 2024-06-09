### Versioning Data after Data Ingestion and Data Preparation Stages

After `DataIngestion` stage is executed the `data` folder get populated by the files downloaded from Kaggle. 
After `DataPreparation` stage is executed a file called `metadataV2.csv` is added to the `data` folder.

Once these steps are complete, we can use DVC to version control the entire data directory using a remote Google Drive folder.

Assuming at this point, DVC is absent from the project - we will set it up from scratch, add our data folder, 
