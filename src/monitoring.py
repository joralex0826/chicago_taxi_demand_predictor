from datetime import datetime, timedelta
from argparse import ArgumentParser

import pandas as pd

import src.config as config
from src.logger import get_logger
from src.config import FEATURE_GROUP_PREDICTIONS_METADATA, FEATURE_GROUP_METADATA
from src.feature_store_api import get_or_create_feature_group, get_feature_store

logger = get_logger()


import pandas as pd
from datetime import datetime, timedelta
from argparse import ArgumentParser


def load_predictions_and_actual_values_from_store(
    from_date: datetime,
    to_date: datetime,
) -> pd.DataFrame:
    """Fetches model predictions and actuals values from
    `from_date` to `to_date` from the Feature Store and returns a dataframe

    Args:
        from_date (datetime): min datetime for which we want predictions and
        actual values

        to_date (datetime): max datetime for which we want predictions and
        actual values

    Returns:
        pd.DataFrame: 4 columns
            - `pickup_location`
            - `predicted_demand`
            - `pickup_hour`
            - `rides`
    """
    # Asegurar que las fechas estén en UTC y con el formato adecuado
    from_date = pd.Timestamp(from_date)
    to_date = pd.Timestamp(to_date)

    if from_date.tzinfo is None:
        from_date = from_date.tz_localize("UTC")
    else:
        from_date = from_date.tz_convert("UTC")
    
    if to_date.tzinfo is None:
        to_date = to_date.tz_localize("UTC")
    else:
        to_date = to_date.tz_convert("UTC")

    # 2 feature groups we need to merge
    predictions_fg = get_or_create_feature_group(FEATURE_GROUP_PREDICTIONS_METADATA)
    actuals_fg = get_or_create_feature_group(FEATURE_GROUP_METADATA)

    # Convertir timestamps a milisegundos
    from_ts = int(from_date.timestamp() * 1000)
    to_ts = int(to_date.timestamp() * 1000)
    print(predictions_fg)
    print(actuals_fg)
    print(actuals_fg.primary_key)

    query = predictions_fg.select_all() \
        .join(
            actuals_fg.select(['pickup_location', 'pickup_hour', 'rides']),
            on=['pickup_hour', 'pickup_location'],
            prefix="actuals_"  # Prefijo para evitar ambigüedad
        ) \
        .filter(predictions_fg.pickup_hour >= from_ts) \
        .filter(predictions_fg.pickup_hour <= to_ts)

    
    # breakpoint()

    # create the feature view `config.FEATURE_VIEW_MONITORING` if it does not exist yet
    feature_store = get_feature_store()
    print(query.show(5))
    print(feature_store)
    try:
        logger.info('Creating monitoring feature view...')

        feature_store.create_feature_view(
            name=config.MONITORING_FV_NAME,
            version=config.MONITORING_FV_VERSION,
            query=query
        )
    except:
        logger.info('Feature view already existed. Skip creation.')

    # feature view
    monitoring_fv = feature_store.get_feature_view(
        name=config.MONITORING_FV_NAME,
        version=config.MONITORING_FV_VERSION
    )
    
    # fetch data from the feature view con márgenes seguros
    monitoring_df = monitoring_fv.get_batch_data(
        start_time=from_date - timedelta(days=7),
        end_time=to_date + timedelta(days=7),
    )

    # Filtrar datos en el rango de tiempo deseado
    monitoring_df = monitoring_df[
        monitoring_df.pickup_hour.between(
            pd.to_datetime(from_ts, unit="ms", utc=True),
            pd.to_datetime(to_ts, unit="ms", utc=True)
        )
    ]

    return monitoring_df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--from_date',
                        type=lambda s: pd.Timestamp(s),
                        help='Datetime argument in the format of YYYY-MM-DD HH:MM:SS')
    parser.add_argument('--to_date',
                        type=lambda s: pd.Timestamp(s),
                        help='Datetime argument in the format of YYYY-MM-DD HH:MM:SS')
    args = parser.parse_args()

    monitoring_df = load_predictions_and_actual_values_from_store(
        from_date=args.from_date,
        to_date=args.to_date
    )
