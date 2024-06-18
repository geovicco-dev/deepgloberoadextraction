from DeepGlobeRoadExtraction.pipeline import DataIngestionPipeline, DataPreparationPipeline, TrainingPipeline, EvaluationPipeline
from DeepGlobeRoadExtraction.utils.common import execute_pipeline

if __name__ == '__main__':
    execute_pipeline('Data Ingestion', DataIngestionPipeline)
    execute_pipeline('Data Preparation', DataPreparationPipeline)
    execute_pipeline('Training', TrainingPipeline)
    execute_pipeline('Evaluation', EvaluationPipeline)