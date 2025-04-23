# 🛠️ Chicago Taxi Demand Predictor

This project provides an end-to-end machine learning pipeline to predict hourly taxi demand in Chicago. It includes a monitoring dashboard for model performance and a real-time map-based prediction interface.

---
## 🔧 Feature Engineering Pipeline

This pipeline is responsible for collecting and transforming raw data into a format suitable for time-series forecasting, then storing it in the **Hopsworks Feature Store**.

**Main tasks:**
1. Retrieve historical taxi ride data for the city of Chicago.
2. Preprocess and align data into hourly intervals.
3. Convert timestamps and format data into a time-series structure.
4. Generate lag features to capture temporal patterns in demand.
5. Save the final feature set to the Hopsworks Feature Store.

## 🧠 Model Training Pipeline

This pipeline trains a predictive model using the preprocessed features from the feature store and registers the model for deployment.

**Key steps:**
1. Load the hourly demand data from the feature store.
2. Define the problem structure:
   - **Target**: Number of rides per hour per location.
   - **Features**: Temporal variables (hour, day, weekday), lag features.
3. Train a regression model using **LightGBM**.
4. Engineer additional features such as:
   - Rolling window statistics to capture trends and seasonality.
5. Validate the model using **Mean Absolute Error (MAE)**.
6. Register the trained model in the **Hopsworks Model Registry** for serving and monitoring.

## 🧠 Inference Pipeline

This pipeline handles the generation of hourly taxi demand forecasts for Chicago using the trained model and the latest features stored in the Hopsworks Feature Store.

**Steps:**

1. Retrieve the most recent feature view from the **Hopsworks Feature Store**.
2. Load the trained model from the **Hopsworks Model Registry**.
3. Generate hourly taxi demand predictions.
4. Compare predictions with actual data to calculate **Mean Absolute Error (MAE)**.
5. The inference logic is deployed as a serverless job via `inference_pipeline.py`, scheduled using **GitHub Actions** to run every hour.



## 🔍 Monitoring Dashboard

The **Monitoring Dashboard** displays the **Mean Absolute Error (MAE)** hour-by-hour, helping track model accuracy and detect potential drift.


➡️ **Try it live**: [Monitoring Dashboard on Streamlit](https://frontendmonitoringpy-8x5arxsy6tsugzucupxb5p.streamlit.app/)

---

## 🚖 Real-Time Prediction Dashboard

The **Taxi Demand Prediction Dashboard** shows predicted taxi demand across Chicago zones for the upcoming hour.


➡️ **Try it live**: [Prediction Dashboard on Streamlit](https://frontendpy-4ozkjpxclnqjj2qlgxty32.streamlit.app/)

---

## 📦 Tech Stack

- **Python 3.10+**
- **Streamlit** – frontend
- **Hopsworks** – feature store, model registry
- **scikit-learn / XGBoost** – modeling
- **GeoPandas, CARTO, PyDeck** – geospatial visualization
- **Poetry** – dependency management
- **GitHub Actions** – CI/CD for testing and deployment

---

## 🛠 Continuous Integration

This project uses **GitHub Actions** for automated testing and quality checks. The workflows include:

- **Linting** with `flake8` to enforce code style
- **Dependency management** with `Poetry`

Workflows are defined in the `.github/workflows/` directory and are triggered on each push and pull request to ensure reliability and maintainability of the codebase.

