import zipfile
from datetime import datetime

import requests
import numpy as np
import pandas as pd

import streamlit as st
import geopandas as gpd
import pydeck as pdk

from src.inference import(
    load_batch_of_features_from_store,
    load_model_from_registry,
    get_model_predictions
)

from src.paths import DATA_DIR
from src.plot import plot_one_sample

st.set_page_config(layout="wide")

current_date = pd.Timestamp.now(tz="UTC").floor("H")
st.title(f'Taxi demand prediction 🚕')
st.header(f'{current_date}')


progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 7

import requests
import zipfile
import geopandas as gpd
from pathlib import Path

# Define el directorio donde se guardarán los datos
DATA_DIR = Path("data")

def load_shape_data_file() -> gpd.GeoDataFrame:
    """
    Descarga y carga los datos geoespaciales de los vecindarios de Chicago.

    Raises:
        Exception: Si no se puede acceder al archivo remoto.

    Returns:
        GeoDataFrame: columnas -> (the_geom, pri_neigh, sec_neigh, geometry)
    """
    # URL del archivo ZIP con los vecindarios de Chicago
    URL = "https://data.cityofchicago.org/api/geospatial/igwz-8jzy?method=export&format=Shapefile"
    
    # Define la ruta del archivo ZIP
    path = DATA_DIR / "chicago_neighborhoods.zip"
    
    # Descarga el archivo
    response = requests.get(URL)
    if response.status_code == 200:
        DATA_DIR.mkdir(parents=True, exist_ok=True)  # Crea la carpeta si no existe
        with open(path, "wb") as f:
            f.write(response.content)
    else:
        raise Exception(f"{URL} is not available")

    # Extrae los archivos
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR / "chicago_neighborhoods")

    # Carga el archivo .shp y lo convierte a EPSG:4326 (coordenadas geográficas estándar)
    shapefile_path = next((DATA_DIR / "chicago_neighborhoods").glob("*.shp"))
    return gpd.read_file(shapefile_path).to_crs("epsg:4326")


with st.spinner(text="Downloading shape file to plot taxi zones"):
    geo_df = load_shape_data_file()
    geo_df['community'] = geo_df['community'].str.lower().str.replace(r"[^a-z0-9\s]", "", regex=True)
    st.sidebar.write('✅ Shape file was downloaded ')
    progress_bar.progress(1/N_STEPS)

with st.spinner(text="Fetching batch of features used in the last run"):
    features_df = load_batch_of_features_from_store(current_date)
    features_df['pickup_location'] = features_df['pickup_location'].str.lower().str.replace(r"[^a-z0-9\s]", "", regex=True)
    st.sidebar.write('✅ Inference features fetched from the store')
    progress_bar.progress(2/N_STEPS)
    print(f'{features_df}')

with st.spinner(text="Loading ML model from registry"):
    model = load_model_from_registry()
    st.sidebar.write('✅ ML model was load from the registry')
    progress_bar.progress(3/N_STEPS)

with st.spinner(text="Computing model predictions"):
    predictions_df = get_model_predictions(model, features_df)
    st.sidebar.write('✅ Model predictions arrived')
    progress_bar.progress(4/N_STEPS)

with st.spinner(text="Preparing data to plot"):

    def pseudocolor(val, minval, maxval, startcolor, stopcolor):
        """
        Convert value in the range minval...maxval to a color in the range
        startcolor to stopcolor. The colors passed and the the one returned are
        composed of a sequence of N component values.

        Credits to https://stackoverflow.com/a/10907855
        """
        f = float(val-minval) / (maxval-minval)
        return tuple(f*(b-a)+a for (a, b) in zip(startcolor, stopcolor))
    print('geooooo')
    print(geo_df)
    df = pd.merge(geo_df, predictions_df, 
                        right_on='pickup_location', 
                        left_on='community', 
                        how='right')
    not_matched = df[df['pickup_location'].isna()]['community']

    print("Valores de 'community' que no encontraron coincidencia:")
    print(not_matched)
    BLACK, GREEN = (0, 0, 0), (0, 255, 0)
    df['color_scaling'] = df['predicted_demand']
    max_pred, min_pred = df['color_scaling'].max(), df['color_scaling'].min()
    df['fill_color'] = df['color_scaling'].apply(lambda x: pseudocolor(x, min_pred, max_pred, BLACK, GREEN))
    progress_bar.progress(5/N_STEPS)

with st.spinner(text="Generating Chicago Map"):

    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=41.8781,   # Latitud de Chicago
        longitude=-87.6298, # Longitud de Chicago
        zoom=11,
        max_zoom=16,
        pitch=45,
        bearing=0
    )

    geojson = pdk.Layer(
        "GeoJsonLayer",
        df,
        opacity=0.25,
        stroked=False,
        filled=True,
        extruded=False,
        wireframe=True,
        get_elevation=10,
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255],
        auto_highlight=True,
        pickable=True,
    )

    tooltip = {"html": "<b>Zone:</b> [{area_num_1}] {community} <br /> <b>Predicted rides:</b> {predicted_demand}"}

    r = pdk.Deck(
        layers=[geojson],
        initial_view_state=INITIAL_VIEW_STATE,
        tooltip=tooltip
    )

    st.pydeck_chart(r)
    progress_bar.progress(6/N_STEPS)

with st.spinner(text="Plotting time-series data"):

    predictions_df = df

    row_indices = np.argsort(predictions_df['predicted_demand'].values)[::-1]
    n_to_plot = 10

    # plot each time-series with the prediction
    for row_id in row_indices[:n_to_plot]:

        # title
        location_id = predictions_df['area_num_1'].iloc[row_id]
        location_name = predictions_df['pickup_location'].iloc[row_id]
        st.header(f'Location ID: {location_id} - {location_name}')

        # plot predictions
        prediction = predictions_df['predicted_demand'].iloc[row_id]
        st.metric(label="Predicted demand", value=int(prediction))
        
        # plot figure
        # generate figure
        fig = plot_one_sample(
            example_id=location_name,
            features=features_df,
            targets=predictions_df['predicted_demand'],
            predictions=pd.Series(predictions_df['predicted_demand']),
            display_title=False,
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)
        
    progress_bar.progress(7/N_STEPS)