from DeepGlobeRoadExtraction.configuration import ConfigurationManager
from DeepGlobeRoadExtraction import logger
from DeepGlobeRoadExtraction.components.training import TrainingComponents

STAGE_NAME = 'Training'

class TrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = ConfigurationManager().get_training_config()
        pipeline = TrainingComponents(config=config)
        pipeline.create_dataloaders()
        pipeline.initialise_model()
        pipeline.create_callbacks()
        pipeline.tune_lr()
        pipeline.create_trainer()
        pipeline.train()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> Processing Stage: {STAGE_NAME} <<<<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> Finished {STAGE_NAME} <<<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    