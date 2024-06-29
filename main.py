from DeepGlobeRoadExtraction import logger
from DeepGlobeRoadExtraction.pipeline import DataIngestionPipeline, DataPreparationPipeline
from DeepGlobeRoadExtraction.utils.common import execute_pipeline

if __name__ == '__main__':
    execute_pipeline('Data Ingestion', DataIngestionPipeline)
    execute_pipeline('DataPreparationPipeline', DataPreparationPipeline)