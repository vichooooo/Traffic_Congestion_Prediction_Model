import os
import sys
from dataclasses import dataclass

# Import your models
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exceptions import CustomException
from src.logger import logging
# --- FIX: Removed the failing import ---
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    # --- This path is now correct ---
    trained_model_file_path = os.path.join("artifacts", "training", "model.joblib")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            
            # The last column is the target
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            
            # --- Model Dictionary ---
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            # --- Optional: Hyperparameters for GridSearch ---
            # You can expand this section
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
            
            # --- Model Evaluation ---
            # Using the evaluate_models function from src/utils.py
            # This assumes you have an evaluate_models function
            
            # If you don't have evaluate_models, you can use this simple loop:
            model_report:dict = {}
            for i in range(len(list(models))):
                model = list(models.values())[i]
                model_name = list(models.keys())[i]
                
                logging.info(f"Training {model_name}...")
                model.fit(X_train, y_train) # Train model
                
                y_test_pred = model.predict(X_test)
                
                test_model_score = r2_score(y_test, y_test_pred)
                
                model_report[model_name] = test_model_score
                logging.info(f"{model_name} R2 Score: {test_model_score}")

            # --- Get Best Model ---
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 score > 0.6")
            
            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            # --- Save the Best Model ---
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"Saved best model to {self.model_trainer_config.trained_model_file_path}")

            return best_model_score
            
        except Exception as e:
            raise CustomException(e, sys)

