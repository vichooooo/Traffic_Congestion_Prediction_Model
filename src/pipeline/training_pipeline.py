import os
import sys
from src.exceptions import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline")
            
            # Step 1: Data Ingestion
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            
            # Step 2: Data Transformation
            train_arr, test_arr, _ = self.data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            
            # Step 3: Model Trainer
            result_message = self.model_trainer.initiate_model_training(train_arr, test_arr)
            
            logging.info(f"Training pipeline completed. {result_message}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    logging.info("Creating training pipeline object.")
    pipeline = TrainPipeline()
    pipeline.run_pipeline()