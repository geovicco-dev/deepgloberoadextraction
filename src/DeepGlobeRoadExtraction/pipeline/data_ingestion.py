from DeepGlobeRoadExtraction.configuration import ConfigurationManager
from DeepGlobeRoadExtraction.components.data_ingestion import DataIngestionComponents
from DeepGlobeRoadExtraction import logger

STAGE_NAME = 'Data Ingestion'

class DataIngestionPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = ConfigurationManager().get_data_ingestion_config()
        pipeline = DataIngestionComponents(config=config)
        pipeline.initialise_kaggle()
        pipeline.download_dataset()
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> Processing Stage: {STAGE_NAME} <<<<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>>>>> Finished {STAGE_NAME} <<<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e