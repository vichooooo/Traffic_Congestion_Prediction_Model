import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import datetime
from datetime import datetime

# --- Placeholder for custom logger and exception ---
# This allows us to re-use your PredictionData class without modification
class _MockLogger:
    def info(self, msg):
        print(f"INFO: {msg}")
    
    def warning(self, msg):
        print(f"WARN: {msg}")
        st.warning(msg)

    def error(self, msg):
        print(f"ERROR: {msg}")
        st.error(msg)

logging = _MockLogger()

class CustomException(Exception):
    def __init__(self, message, error_detail):
        super().__init__(message)
        print(f"CustomException: {message}, Detail: {error_detail}")

# --------------------------------------------------

# --- Setup Paths ---
# Get the absolute path of the directory where this script is (which is /src)
base_dir = os.path.dirname(os.path.abspath(__file__))

# The models are in the SAME directory as the script, so we just join the base_dir
# with the filenames.
model_path = os.path.join(base_dir, "model.joblib")
preprocessor_path = os.path.join(base_dir, "preprocessor.joblib")

# --- Load Model and Preprocessor (Cached) ---
@st.cache_resource
def load_artifacts():
    """
    Loads the model and preprocessor from disk.
    Uses Streamlit's caching to load only once.
    """
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        logging.info(f"Successfully loaded model and preprocessor from {artifacts_path}")
        return model, preprocessor
    except Exception as e:
        logging.warning(f"Error loading model/preprocessor: {e}")
        logging.warning("Running in PLACEHOLDER MODE. Predictions will be random.")
        return None, None

model, preprocessor = load_artifacts()

# --- Custom Prediction Class (Copied from your Flask app) ---
# This class will hold the data from the form
class PredictionData:
    def __init__(self,
                 temp: float,
                 rain_1h: float,
                 snow_1h: float,
                 clouds_all: int,
                 holiday: str,
                 weather: str,
                 weather_description: str,
                 date: str,
                 time: str):
        
        self.temp = temp
        self.rain_1h = rain_1h
        self.snow_1h = snow_1h
        self.clouds_all = clouds_all
        self.holiday = holiday
        self.weather = weather # This is from the form, will be mapped to 'weather_main'
        self.weather_description = weather_description
        self.date = date
        self.time = time

    def get_data_as_dataframe(self):
        try:
            
            # --- THIS IS THE FIX ---
            # 1. Create a temporary 'date_time' field from the form inputs
            # The format %Y-%m-%d %H:%M is what the HTML form provides
            # Note: We already formatted the st.date_input and st.time_input to this format
            temp_datetime = datetime.strptime(f"{self.date} {self.time}", "%Y-%m-%d %H:%M")

            # 2. Manually create the engineered features that the preprocessor expects
            hour = temp_datetime.hour
            month = temp_datetime.month
            day_of_week = temp_datetime.weekday() # Use .weekday() for 0-6
            is_weekend = 1 if day_of_week >= 5 else 0 # Creates the 'is_weekend' column

            # 3. Create the dictionary with all columns the preprocessor was trained on
            #    The names here *must* match your 'numerical_cols' and 'categorical_cols'
            
            data_dict = {
                # Numerical columns
                "temp": [self.temp],
                "rain_1h": [self.rain_1h],
                "snow_1h": [self.snow_1h],
                "clouds_all": [self.clouds_all],
                "hour": [hour],
                "month": [month],
                "day_of_week": [day_of_week],
                "is_weekend": [is_weekend], # The missing column is now added
                
                # Categorical columns
                "holiday": [self.holiday],
                "weather_main": [self.weather], # Maps form 'weather' to 'weather_main'
                "weather_description": [self.weather_description]
            }
            # ---------------------
            
            df = pd.DataFrame(data_dict)
            logging.info("Dataframe for prediction created successfully")
            logging.info(f"DataFrame head: \n{df.head()}")
            return df
            
        except Exception as e:
            logging.error(f"Error in get_data_as_dataframe: {e}")
            raise CustomException(e, sys)

# --- Streamlit App UI ---
st.set_page_config(page_title="Traffic Volume Predictor", layout="wide")
st.title("🚗 Metro Interstate Traffic Volume Predictor")

# Check if we are in placeholder mode
if model is None or preprocessor is None:
    st.error("Model and/or preprocessor files not found. The app cannot make predictions.")
else:
    st.info("Model and preprocessor loaded successfully. Please enter the details below.")

    # --- Input Options ---
    # These are assumptions based on the UCI dataset. Update them as needed.
    holiday_options = [
        'None', 'Martin Luther King Jr Day', 'Washingtons Birthday', 'Memorial Day',
        'Independence Day', 'Labor Day', 'Columbus Day', 'Veterans Day',
        'Thanksgiving Day', 'Christmas Day', 'New Years Day', 'State Fair'
    ]
    
    weather_main_options = [
        'Clear', 'Clouds', 'Rain', 'Snow', 'Mist', 'Drizzle', 'Haze',
        'Fog', 'Thunderstorm', 'Squall', 'Smoke', 'Sand', 'Dust', 'Ash', 'Tornado'
    ]
    
    # A sample of weather descriptions
    weather_desc_options = [
        'sky is clear', 'few clouds', 'scattered clouds', 'broken clouds', 'overcast clouds',
        'light rain', 'moderate rain', 'heavy intensity rain', 'very heavy rain',
        'light snow', 'Snow', 'heavy snow', 'Mist', 'Fog', 'Haze', 'light intensity shower rain',
        'proximity thunderstorm', 'thunderstorm with light rain', 'thunderstorm with heavy rain'
    ]

    # --- Prediction Form ---
    with st.form("prediction_form"):
        st.header("Enter Conditions to Predict Traffic")
        
        # --- Date and Time ---
        col1, col2 = st.columns(2)
        with col1:
            date_val = st.date_input("Date", datetime.now())
        with col2:
            time_val = st.time_input("Time", datetime.now().time())
        
        # --- Weather Metrics ---
        st.subheader("Weather Conditions")
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            # Assuming temp is in Kelvin as per the UCI dataset
            temp_val = st.number_input("Temperature (Kelvin)", min_value=250.0, max_value=320.0, value=288.15, step=0.1)
        with col4:
            rain_val = st.number_input("Rain (mm in last 1h)", min_value=0.0, value=0.0, step=0.1)
        with col5:
            snow_val = st.number_input("Snow (mm in last 1h)", min_value=0.0, value=0.0, step=0.1)
        with col6:
            clouds_val = st.number_input("Clouds (% sky cover)", min_value=0, max_value=100, value=40, step=1)

        # --- Categorical Inputs ---
        st.subheader("Event and Weather Type")
        col7, col8, col9 = st.columns(3)
        with col7:
            holiday_val = st.selectbox("Holiday", options=holiday_options, index=0)
        with col8:
            weather_val = st.selectbox("Main Weather", options=weather_main_options, index=1)
        with col9:
            # Using selectbox for consistency, but text_input is also an option
            weather_desc_val = st.selectbox("Weather Description", options=weather_desc_options, index=3)

        # --- Submit Button ---
        submitted = st.form_submit_button("Predict Traffic Volume")

    # --- Prediction Logic ---
    if submitted:
        try:
            # 1. Format date and time inputs for the PredictionData class
            date_str = date_val.strftime("%Y-%m-%d")
            time_str = time_val.strftime("%H:%M")

            # 2. Create PredictionData object
            pred_data = PredictionData(
                temp=float(temp_val),
                rain_1h=float(rain_val),
                snow_1h=float(snow_val),
                clouds_all=int(clouds_val),
                holiday=holiday_val,
                weather=weather_val,
                weather_description=weather_desc_val,
                date=date_str,
                time=time_str
            )
            
            # 3. Get data as a DataFrame
            pred_df = pred_data.get_data_as_dataframe()

            # 4. Make prediction
            logging.info("Applying preprocessor...")
            data_processed = preprocessor.transform(pred_df)
            
            logging.info("Making prediction...")
            prediction = model.predict(data_processed)
            
            output = round(prediction[0], 2)
            
            logging.info(f"Prediction successful: {output}")

            # 5. Display the result
            st.metric("Predicted Traffic Volume", f"{int(output)} vehicles/hour")
            
            with st.expander("Show Features Sent to Model"):
                st.write("These are the raw and engineered features created from your inputs:")
                st.dataframe(pred_df)

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            # The custom logger will also call st.error to show it on the UI