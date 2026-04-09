import os
import sys
import pandas as pd
import numpy as np
import json
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from dataclasses import dataclass

from src.exceptions import CustomException
from src.logger import logging
from src.utils import load_object

@dataclass
class ModelEvaluationConfig:
    metrics_file_path: str = os.path.join("artifacts", "metrics.json")

class ModelEvaluation:
    def __init__(self):
        self.evaluation_config = ModelEvaluationConfig()

    def _calculate_metrics(self, y_true, y_pred):
        '''Calculates regression metrics'''
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return r2, mae, rmse

    def initiate_model_evaluation(self, test_array):
        try:
            logging.info("Starting model evaluation component")
            
            # Split test array into features and target
            X_test, y_test = (
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define paths and load the saved model
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(file_path=model_path)
            
            if model is None:
                logging.error("Model file not found. Aborting evaluation.")
                raise CustomException("Model file (model.pkl) not found", sys)

            logging.info("Model loaded successfully for evaluation")

            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Calculate metrics
            (r2, mae, rmse) = self._calculate_metrics(y_test, y_pred)

            logging.info("Model Evaluation Metrics:")
            logging.info(f"  R2 Score: {r2:.4f}")
            logging.info(f"  Mean Absolute Error (MAE): {mae:.4f}")
            logging.info(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")

            # Save metrics to a JSON file
            metrics = {
                "r2_score": r2,
                "mean_absolute_error": mae,
                "root_mean_squared_error": rmse
            }
            with open(self.evaluation_config.metrics_file_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logging.info(f"Metrics saved to {self.evaluation_config.metrics_file_path}")

            return r2, mae, rmse

        except Exception as e:
            raise CustomException(e, sys)