from tqdm import tqdm
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.wkt import loads
from src.paths import RAW_DATA_DIR
from typing import Tuple


def fetch_chicago_data():
    """
    Fetches Chicago data from a file.
    """
    rides = pd.read_parquet(RAW_DATA_DIR / 'taxi_trips.parquet')
    
    return rides

def validate_raw_data(rides: pd.DataFrame) -> pd.DataFrame:
    rides['pickup_datetime'] = pd.to_datetime(rides['pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

    if rides['pickup_datetime'].isnull().any():
        print("Warning: There are null values in 'pickup_datetime'.")
        rides = rides.dropna(subset=['pickup_datetime'])

    min_date = rides['pickup_datetime'].min()
    max_date = rides['pickup_datetime'].max()
    print(f"Date range: {min_date} to {max_date}")
    
    expected_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    missing_dates = expected_dates.difference(rides['pickup_datetime'].dt.date.unique())
    if not missing_dates.empty:
        print(f"Warning: Missing data for the following days: {missing_dates}")
    
    return rides

def load_raw_data() -> pd.DataFrame:
    """
    Loads raw data from local storage or downloads it from the Chicago website, and
    then loads it into a Pandas DataFrame

    Returns:
        pd.DataFrame: DataFrame with the following columns:
            - pickup_datetime: datetime of the pickup
            - pickup_longitude: pickup location longitude
            - pickup_latitude: pickup location latitude
    """  
    rides = fetch_chicago_data()
    rides = rides.copy()[['Trip Start Timestamp', 'Pickup Centroid Latitude', 'Pickup Centroid Longitude']]
    rides.rename(columns={'Trip Start Timestamp':'pickup_datetime',
                            'Pickup Centroid Latitude':'pickup_latitude',
                            'Pickup Centroid Longitude':'pickup_longitude'}, inplace=True)

    # validate the file
    rides_validated = validate_raw_data(rides)

    if rides_validated.empty:
        # no data, so we return an empty dataframe
        return pd.DataFrame()
    else:
        # keep only time and origin of the ride
        rides_validated = rides_validated[['pickup_datetime', 'pickup_latitude', 'pickup_longitude']]
        return rides_validated
    
def load_geo_data():
    chicago_zones = pd.read_parquet(RAW_DATA_DIR / 'chicago_geo_data.parquet')
    chicago_zones["geometry"] = chicago_zones["the_geom"].apply(loads)
    chicago_zones = gpd.GeoDataFrame(chicago_zones, geometry="geometry", crs="EPSG:4326")
    return chicago_zones

def merge_geo_and_ts_data(rides):
    chicago_zones = load_geo_data()
    geometry = [Point(xy) for xy in zip(rides['pickup_longitude'], rides['pickup_latitude'])]
    rides_gdf = gpd.GeoDataFrame(rides, geometry=geometry, crs="EPSG:4326")
    rides_with_zones = gpd.sjoin(rides_gdf, chicago_zones[['PRI_NEIGH', 'geometry']], 
                                how="left", predicate="within")

    rides_with_zones = rides_with_zones[rides.columns.tolist() + ['PRI_NEIGH']]
    rides_with_zones['pickup_location'] = rides_with_zones['PRI_NEIGH'].fillna("Outside Chicago")
    rides_with_zones.drop(columns='PRI_NEIGH', inplace=True)
    return rides_with_zones

def add_missing_slots(ts_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add necessary rows to the input 'ts_data' to make sure the output
    has a complete list of:
    - pickup_hours
    - pickup_location
    - pickup_longitude
    - pickup_latitude
    """
    locations = ts_data['pickup_location'].unique()
    full_range = pd.date_range(
        ts_data['pickup_hour'].min(), ts_data['pickup_hour'].max(), freq='h'
    )
    location_coords = (
        ts_data[['pickup_location', 'pickup_longitude', 'pickup_latitude']]
        .dropna()
        .groupby('pickup_location')
        .first()
        .to_dict(orient='index')
    )

    output = pd.DataFrame()
    for location in tqdm(locations):
        ts_data_i = ts_data.loc[
            ts_data.pickup_location == location,
            ['pickup_hour', 'rides', 'pickup_longitude', 'pickup_latitude']
        ].groupby('pickup_hour', as_index=False).first()

        if location == "Outside Chicago":
            pickup_longitude, pickup_latitude = 0.0, 0.0
        else:
            coords = location_coords.get(location, {'pickup_longitude': 0.0, 'pickup_latitude': 0.0})
            pickup_longitude, pickup_latitude = coords['pickup_longitude'], coords['pickup_latitude']

        if ts_data_i.empty:
            ts_data_i = pd.DataFrame.from_dict([
                {'pickup_hour': ts_data['pickup_hour'].max(), 'rides': 0, 
                 'pickup_longitude': pickup_longitude, 'pickup_latitude': pickup_latitude}
            ])

        ts_data_i.set_index('pickup_hour', inplace=True)
        ts_data_i.index = pd.DatetimeIndex(ts_data_i.index)
        ts_data_i = ts_data_i.reindex(full_range, fill_value=0)
        ts_data_i['pickup_longitude'] = pickup_longitude
        ts_data_i['pickup_latitude'] = pickup_latitude
        ts_data_i['pickup_location'] = location

        output = pd.concat([output, ts_data_i])

    output = output.reset_index().rename(columns={'index': 'pickup_hour'})
    output.dropna(inplace=True)

    return output

def transform_raw_data_into_ts_data(rides: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw data into time-series data
    """
    rides['pickup_hour'] = rides['pickup_datetime'].dt.round('h')
    location_coords = rides.groupby('pickup_location')[['pickup_longitude', 'pickup_latitude']].first().reset_index()
    location_coords.loc[location_coords['pickup_location'] == 'Outside Chicago', ['pickup_longitude', 'pickup_latitude']] = (0.0, 0.0)
    agg_rides = rides.groupby(['pickup_hour', 'pickup_location']).size().reset_index(name='rides')
    agg_rides = agg_rides.merge(location_coords, on='pickup_location', how='left')
    agg_rides_all_slots = add_missing_slots(agg_rides)
    agg_rides_all_slots.dropna(inplace=True)

    return agg_rides_all_slots

def get_cutoff_indices_features_and_target(
    data: pd.DataFrame,
    input_seq_len: int,
    step_size: int
    ) -> list:

        stop_position = len(data) - 1
        
        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0
        subseq_mid_idx = input_seq_len
        subseq_last_idx = input_seq_len + 1
        indices = []
        
        while subseq_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
            subseq_first_idx += step_size
            subseq_mid_idx += step_size
            subseq_last_idx += step_size

        return indices


def transform_ts_data_into_features_and_target(
    ts_data: pd.DataFrame,
    input_seq_len: int,
    step_size: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Slices and transposes data from time-series format into a (features, target)
    format that we can use to train Supervised ML models
    """
    assert set(['pickup_hour', 'rides', 'pickup_location', 'pickup_longitude', 'pickup_latitude']).issubset(ts_data.columns)

    locations = ts_data['pickup_location'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()
    
    for location in tqdm(locations):
        
        # keep only ts data for this `location`
        ts_data_one_location = ts_data.loc[
            ts_data.pickup_location == location, 
            ['pickup_hour', 'rides', 'pickup_longitude', 'pickup_latitude']
        ].sort_values(by=['pickup_hour'])

        # Obtener latitud y longitud (se toma el primer valor ya que es constante por ubicación)
        pickup_longitude = ts_data_one_location['pickup_longitude'].iloc[0]
        pickup_latitude = ts_data_one_location['pickup_latitude'].iloc[0]

        # pre-compute cutoff indices to split dataframe rows
        indices = get_cutoff_indices_features_and_target(
            ts_data_one_location,
            input_seq_len,
            step_size
        )

        # slice and transpose data into numpy arrays for features and targets
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
        y = np.ndarray(shape=(n_examples), dtype=np.float32)
        pickup_hours = []
        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['rides'].values[0]
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

        # numpy -> pandas
        features_one_location = pd.DataFrame(
            x,
            columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
        )
        features_one_location['pickup_hour'] = pickup_hours
        features_one_location['pickup_location'] = location
        features_one_location['pickup_longitude'] = pickup_longitude
        features_one_location['pickup_latitude'] = pickup_latitude

        # numpy -> pandas
        targets_one_location = pd.DataFrame(y, columns=['target_rides_next_hour'])

        # concatenate results
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])

    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)

    return features, targets['target_rides_next_hour']
