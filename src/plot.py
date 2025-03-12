from typing import Optional, List
from datetime import timedelta

import pandas as pd
import plotly.express as px 

def plot_one_sample(
    example_id: str,  # Pickup location is a string
    features: pd.DataFrame,
    targets: Optional[pd.Series] = None,
    predictions: Optional[pd.Series] = None,
    display_title: Optional[bool] = True,
):
    # Buscar el índice numérico de la fila que corresponde a `example_id` en la columna 'pickup_location'
    example_index = features[features['pickup_location'] == example_id].index[0]
    
    features_ = features.iloc[example_index]
    
    if targets is not None:
        target_ = targets.iloc[example_index]
    else:
        target_ = None
    
    ts_columns = [c for c in features.columns if c.startswith('rides_previous_')]
    ts_values = [features_[c] for c in ts_columns] + [target_]
    ts_dates = pd.date_range(
        features_['pickup_hour'] - timedelta(hours=len(ts_columns)),
        features_['pickup_hour'],
        freq='h'
    )
    
    # Line plot with past values
    title = f'Pick up hour={features_["pickup_hour"]}, location_id={features_["pickup_location"]}' if display_title else None
    fig = px.line(
        x=ts_dates, y=ts_values,
        template='plotly_dark',
        markers=True, title=title
    )
    
    if targets is not None:
        # Green dot for the value we want to predict
        fig.add_scatter(x=ts_dates[-1:], y=[target_],
                        line_color='green',
                        mode='markers', marker_size=10, name='actual value') 
        
    if predictions is not None:
        # Big red X for the predicted value, if passed
        prediction_ = predictions.iloc[example_index]
        fig.add_scatter(x=ts_dates[-1:], y=[prediction_],
                        line_color='red',
                        mode='markers', marker_symbol='x', marker_size=15,
                        name='prediction')             

    return fig



def plot_ts(
    ts_data: pd.DataFrame,
    locations: Optional[List[int]] = None
    ):
    """
    Plot time-series data
    """
    ts_data_to_plot = ts_data[ts_data.pickup_location.isin(locations)] if locations else ts_data

    fig = px.line(
        ts_data_to_plot,
        x="pickup_hour",
        y="rides",
        color='pickup_location',
        template='none',
    )

    fig.show()