import sys
import os
import pandas as pd
import numpy as np
from src.exceptions import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        # Point to the artifacts (or production_model) folder
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        logging.info("PredictPipeline initialized.")

    def predict(self, features):
        try:
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            logging.info("Model and preprocessor loaded successfully for prediction.")
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            logging.info("Prediction complete.")
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 holiday: str,
                 temp: float,
                 rain_1h: float,
                 snow_1h: float,
                 clouds_all: int,
                 weather_main: str,
                 weather_description: str,
                 date_time: str):

        self.holiday = holiday
        self.temp = temp
        self.rain_1h = rain_1h
        self.snow_1h = snow_1h
        self.clouds_all = clouds_all
        self.weather_main = weather_main
        self.weather_description = weather_description
        self.date_time = date_time
        logging.info("CustomData object created.")

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "holiday": [self.holiday],
                "temp": [self.temp],
                "rain_1h": [self.rain_1h],
                "snow_1h": [self.snow_1h],
                "clouds_all": [self.clouds_all],
                "weather_main": [self.weather_main],
                "weather_description": [self.weather_description],
                "date_time": [self.date_time],
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Custom data DataFrame created.")

            df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M:%S')
            df['hour'] = df['date_time'].dt.hour
            df['month'] = df['date_time'].dt.month
            df['day_of_week'] = df['date_time'].dt.dayofweek
            df['is_weekend'] = np.where(df['day_of_week'] >= 5, 1, 0)
            
            df = df.drop(['date_time'], axis=1)
            logging.info("Feature engineering on custom data complete.")

            return df
        
        except Exception as e:
            raise CustomException(e, sys)