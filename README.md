# 🚗 Metro Interstate Traffic Congestion Prediction

A machine learning web application that predicts traffic volume on metro interstate highways based on weather conditions, time of day, and holiday information. Built with Streamlit and trained on the UCI Metro Interstate Traffic Volume dataset.

---

## 📌 Project Overview

This project implements an end-to-end ML pipeline — from data ingestion and transformation to model training and deployment — wrapped in an interactive Streamlit web interface where users can input real-world conditions and get an instant traffic volume prediction.

---

## 🚀 Demo

Run the app locally:
```bash
streamlit run app.py
```

---

## 🗂️ Project Structure

```
Traffic_Congestion_Prediction_Model/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
│
├── artifacts/
│   ├── raw_data.csv                # Original dataset
│   ├── train.csv                   # Training split
│   ├── test.csv                    # Test split
│   └── training/
│       ├── model.joblib            # Trained model
│       └── preprocessor.joblib    # Fitted preprocessor
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Data loading and splitting
│   │   ├── data_transformation.py # Feature engineering and preprocessing
│   │   ├── model_trainer.py        # Model training and selection
│   │   ├── model_evaluation.py     # Model evaluation metrics
│   │   └── model_pusher.py         # Save and export model
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py    # End-to-end training pipeline
│   │   └── prediction_pipeline.py  # Inference pipeline
│   │
│   ├── logger.py                   # Custom logging
│   ├── exceptions.py               # Custom exception handling
│   └── utils.py                    # Helper functions
│
├── notebook/
│   └── Traffic_Congestion_Prediction.ipynb  # EDA and experimentation
│
└── templates/
    └── traffic_app.html            # HTML template
```

---

## 🧠 Models Evaluated

The training pipeline evaluates multiple regression models and selects the best performer based on R² score:

| Model | Notes |
|---|---|
| Random Forest | ✅ Ensemble method |
| Decision Tree | Base learner |
| Gradient Boosting | Boosting ensemble |
| XGBoost | Optimized boosting |
| CatBoost | Handles categoricals natively |
| AdaBoost | Adaptive boosting |
| Linear Regression | Baseline |

The best model (R² > 0.6) is automatically saved to `artifacts/training/model.joblib`.

---

## 🔧 Features Used

| Feature | Type | Description |
|---|---|---|
| `temp` | Numerical | Temperature in Kelvin |
| `rain_1h` | Numerical | Rain volume in last 1 hour (mm) |
| `snow_1h` | Numerical | Snow volume in last 1 hour (mm) |
| `clouds_all` | Numerical | Cloud coverage (%) |
| `hour` | Numerical | Hour of day (engineered) |
| `month` | Numerical | Month of year (engineered) |
| `day_of_week` | Numerical | Day of week (engineered) |
| `is_weekend` | Numerical | Weekend flag (engineered) |
| `holiday` | Categorical | US public holiday name or 'None' |
| `weather_main` | Categorical | Main weather condition |
| `weather_description` | Categorical | Detailed weather description |

---

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/vichooooo/Traffic_Congestion_Prediction_Model.git
cd Traffic_Congestion_Prediction_Model
```

2. **Create a virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the app:**
```bash
streamlit run app.py
```

---

## 📦 Dependencies

```
streamlit
scikit-learn
pandas
numpy
xgboost
catboost
matplotlib
seaborn
```

---

## 📊 Dataset

The project uses the [Metro Interstate Traffic Volume dataset](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume) from the UCI Machine Learning Repository. It contains hourly traffic volume data for a stretch of I-94 westbound, along with weather and holiday information from 2012 to 2018.

---

## 👤 Author

**vichooooo** — [GitHub](https://github.com/vichooooo)
