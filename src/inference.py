from datetime import datetime, timedelta
import hopsworks
from hsfs.feature_store import FeatureStore
import pandas as pd
import numpy as np

import src.config as config

def get_hopsworks_project() -> hopsworks.project.Project:

    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

def get_feature_store() -> FeatureStore:

    project = get_hopsworks_project()
    return project.get_feature_store()

def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    features_columns = [c for c in features.columns if c.startswith('rides_')]
    features_to_predict = features[features_columns]
    predictions = model.predict(features_to_predict)
    results = pd.DataFrame()
    results['pickup_location'] = features['pickup_location'].values
    results['predicted_demand'] = predictions.round(0)

    return results

def load_batch_of_features_from_store(
    current_date: datetime,
) -> pd.DataFrame:

    feature_store = get_feature_store()

    n_features = config.N_FEATURES

    # read time-series data from the feature store
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=28)
    print(f'Fetching data from {fetch_data_from} to {fetch_data_to}')
    
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )

    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1)),
    )

    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]

    # validate we are not missing data in the feature store
    location_ids = ts_data['pickup_location'].unique()
    assert len(ts_data) == n_features * len(location_ids), "Time-series data is not complete"

    # sort data by location and time
    ts_data.sort_values(by=['pickup_location', 'pickup_hour'], inplace=True)

    print(f'ts_data: {ts_data}')
    # validate we are not missing data in the feature store
    location_ids = ts_data['pickup_location'].unique()
    assert len(ts_data) == config.N_FEATURES * len(location_ids), \
        "Time-series data is not complete. Make sure your feature pipeline is up and runnning."

    # transpose time-series data as a feature vector, for each `pickup_location`
    x = np.ndarray(shape=(len(location_ids), n_features), dtype=np.float32)
    for i, location_id in enumerate(location_ids):
        ts_data_i = ts_data.loc[ts_data.pickup_location == location_id, :]
        ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
        x[i, :] = ts_data_i['rides'].values

    # numpy arrays to Pandas dataframes
    features = pd.DataFrame(
        x,
        columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(n_features))]
    )
    features['pickup_hour'] = current_date
    features['pickup_location'] = location_ids
    features.sort_values(by=['pickup_location'], inplace=True)

    return features

def load_model_from_registry():
    import joblib
    from pathlib import Path

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name=config.MODEL_NAME,
        version=config.MODEL_VERSION
    )

    model_dir = model.download()
    model = joblib.load(Path(model_dir) / 'model.pkl')

    return model