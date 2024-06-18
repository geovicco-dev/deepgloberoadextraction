from DeepGlobeRoadExtraction.configuration import ConfigurationManager
from DeepGlobeRoadExtraction import logger
from DeepGlobeRoadExtraction.components.data_preparation import DataPreparationComponents

STAGE_NAME = 'Data Preparation'

class DataPreparationPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = ConfigurationManager().get_data_preparation_config()
        pipeline = DataPreparationComponents(config=config)
        pipeline.load_metadata()
        pipeline.split_dataset()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> Processing Stage: {STAGE_NAME} <<<<<<<<")
        obj = DataPreparationPipeline()
        obj.main()
        logger.info(f">>>>>>> Finished {STAGE_NAME} <<<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    