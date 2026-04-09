import os
import sys
import shutil
from dataclasses import dataclass

from src.exceptions import CustomException
from src.logger import logging

@dataclass
class ModelPusherConfig:
    # Path to the trained model and preprocessor
    trained_model_path: str = os.path.join("artifacts", "model.pkl")
    trained_preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    
    # Path where we want to "push" or deploy them
    production_model_dir: str = os.path.join("production_model")
    production_model_path: str = os.path.join(production_model_dir, "model.pkl")
    production_preprocessor_path: str = os.path.join(production_model_dir, "preprocessor.pkl")

class ModelPusher:
    def __init__(self):
        self.pusher_config = ModelPusherConfig()

    def initiate_model_pusher(self):
        try:
            logging.info("Model Pusher component started")

            # Ensure the trained models exist
            if not os.path.exists(self.pusher_config.trained_model_path):
                raise CustomException("Trained model not found in artifacts", sys)
            if not os.path.exists(self.pusher_config.trained_preprocessor_path):
                raise CustomException("Trained preprocessor not found in artifacts", sys)

            # Create the production directory if it doesn't exist
            os.makedirs(self.pusher_config.production_model_dir, exist_ok=True)
            logging.info(f"Production directory created at: {self.pusher_config.production_model_dir}")

            # Copy the files
            shutil.copy(self.pusher_config.trained_model_path, self.pusher_config.production_model_path)
            shutil.copy(self.pusher_config.trained_preprocessor_path, self.pusher_config.production_preprocessor_path)

            logging.info(f"Model pushed to {self.pusher_config.production_model_path}")
            logging.info(f"Preprocessor pushed to {self.pusher_config.production_preprocessor_path}")
            logging.info("Model Pusher component completed successfully")

            return (
                self.pusher_config.production_model_path,
                self.pusher_config.production_preprocessor_path
            )

        except Exception as e:
            raise CustomException(e, sys)