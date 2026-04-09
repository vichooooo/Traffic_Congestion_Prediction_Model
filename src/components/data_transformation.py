import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # --- This is the correct path that app.py needs ---
    preprocessor_obj_file_path = os.path.join('artifacts', 'training', "preprocessor.joblib")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def _feature_engineer(self, df):
        '''
        This function performs feature engineering on the date_time column
        '''
        try:
            # --- FIX for UserWarning: Specify the exact format ---
            df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M')
            
            df['hour'] = df['date_time'].dt.hour
            df['month'] = df['date_time'].dt.month
            df['day_of_week'] = df['date_time'].dt.dayofweek
            df['is_weekend'] = np.where(df['day_of_week'] >= 5, 1, 0)
            
            # Drop the original date_time and the classification target 'Intensity'
            df = df.drop(['date_time', 'Intensity'], axis=1, errors='ignore')
            
            return df
        
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        '''
        This function is responsible for creating the data transformation pipeline
        '''
        try:
            # Define which columns are numerical and which are categorical
            numerical_cols = [
                'temp', 'rain_1h', 'snow_1h', 'clouds_all', 
                'hour', 'month', 'day_of_week', 'is_weekend'
            ]
            
            categorical_cols = [
                'holiday', 'weather_main', 'weather_description'
            ]

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)) # Use with_mean=False for sparse data
                ]
            )

            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Numerical columns: {numerical_cols}")

            # Combine pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols)
                ],
                remainder='passthrough' # Keep any other columns
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            
            # Perform feature engineering
            logging.info("Applying feature engineering")
            train_df = self._feature_engineer(train_df)
            test_df = self._feature_engineer(test_df)

            logging.info("Obtaining preprocessing object")
            preprocessor_obj = self.get_data_transformer_object()

            # Define the target column
            target_column_name = "traffic_volume"

            # Separate features (X) and target (y)
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Apply the preprocessor
            # --- THE FIX: Add .toarray() to convert sparse matrix to dense array ---
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df).toarray()
            # ----------------------------------------------------------------------
        
            # This try/except block is for combining the arrays
            try:
                # Combine features and target back into an array
                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]
                test_arr = np.c_[
                    input_feature_test_arr, np.array(target_feature_test_df)
                ]
                logging.info("--- np.c_ complete ---")

            except Exception as e:
                # This logging helps if the error persists
                logging.error(f"Error in np.c_: {e}")
                logging.error(f"Shape of train_features: {input_feature_train_arr.shape}")
                logging.error(f"Shape of train_target: {np.array(target_feature_train_df).shape}")
                raise CustomException(e, sys)

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Saved preprocessing object.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

