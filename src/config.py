import os

from dotenv import load_dotenv
from src.paths import PARENT_DIR
from src.feature_store_api import FeatureGroupConfig, FeatureViewConfig

# load key-value pairs from .env file located in the parent directory
load_dotenv(PARENT_DIR / '.env')

try:
    HOPSWORKS_PROJECT_NAME = os.environ['HOPSWORKS_PROJECT_NAME']
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception(
        'Create an .env file on the project root with the HOPSWORKS_PROJECT_NAME and HOPSWORKS_API_KEY'
    )

FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 1
FEATURE_GROUP_METADATA = FeatureGroupConfig(
    name='time_series_hourly_feature_group',
    version=1,
    description='Feature group with hourly time-series data of historical taxi rides',
    primary_key=['pickup_location', 'pickup_hour'],
    event_time='pickup_hour',
    online_enabled=True,
)

FEATURE_VIEW_NAME = 'time_series_hourly_feature_view'
FEATURE_VIEW_VERSION = 1

MODEL_NAME = 'taxi_demand_predictor_next_hour'
MODEL_VERSION = 1
N_FEATURES = 24 * 28

FEATURE_GROUP_MODEL_PREDICTIONS = 'model_predictions_feature_group'

FEATURE_GROUP_PREDICTIONS_METADATA = FeatureGroupConfig(
    name='model_predictions_feature_group',
    version=1,
    description='Predictions generate by our production model',
    primary_key=['pickup_location', 'pickup_hour'],
    event_time='pickup_hour',
)

MONITORING_FV_NAME = 'monitoring_feature_view'
MONITORING_FV_VERSION = 1

FEATURE_VIEW_PREDICTIONS_METADATA = FeatureViewConfig(
    name='model_predictions_feature_view',
    version=1,
    feature_group=FEATURE_GROUP_PREDICTIONS_METADATA,
)