from DeepGlobeRoadExtraction.configuration import ConfigurationManager
from DeepGlobeRoadExtraction import logger
from DeepGlobeRoadExtraction.components.evaluation import EvaluationComponents

STAGE_NAME = 'Evaluation'

class EvaluationPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = ConfigurationManager().get_evaluation_config()
        pipeline = EvaluationComponents(config=config)
        pipeline.create_dataloaders()
        pipeline.load_model()
        pipeline.create_trainer()
        pipeline.evaluate()
        pipeline.save_predictions()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> Processing Stage: {STAGE_NAME} <<<<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>> Finished {STAGE_NAME} <<<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    