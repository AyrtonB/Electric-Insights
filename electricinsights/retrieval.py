# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01-retrieval.ipynb (unless otherwise specified).

__all__ = ['query_API', 'dict_col_to_cols', 'clean_nested_dict_cols', 'set_dt_idx', 'create_df_dt_rng', 'clean_df_dts',
           'retrieve_stream_df', 'check_streams', 'retrieve_streams_df', 'get_EI_data', 'bulk_retrieval',
           'get_EI_files', 'check_for_gappy_data', 'retrieve_latest_data', 'year_to_resource', 'construct_datapackage',
           'app', 'download']

# Cell
import pandas as pd
import numpy as np

import os
import json
import typer
import requests
import xmltodict
from datetime import date
from warnings import warn
from tqdm import tqdm

# Cell
def query_API(start_date:str, end_date:str, stream:str, time_group='30m'):
    """
    'Query API' makes the call to Electric Insights and returns the JSON response

    Parameters:
        start_date: Start date for data given as a string in the form '%Y-%m-%d'
        end_date: End date for data given as a string in the form '%Y-%m-%d'
        stream: One of 'prices_ahead', 'prices', 'temperatures', 'emissions', or 'generation-mix'
        time_group: One of '30m', '1h', '1d' or '7d'. The default is '30m'
    """

    # Checking stream is an EI endpoint
    possible_streams = ['prices_ahead', 'prices', 'temperatures', 'emissions', 'generation-mix']
    assert stream in possible_streams, f"Stream must be one of {''.join([stream+', ' for stream in possible_streams])[:-2]}"

    # Checking time_group will be accepted by API
    possible_time_groups = ['30m', '1h', '1d', '7d']
    assert time_group in possible_time_groups, f"Time group must be one of {''.join([time_group+', ' for time_group in possible_time_groups])[:-2]}"

    # Formatting dates
    format_dt = lambda dt: date.strftime(dt, '%Y-%m-%d') if isinstance(dt, date) else dt
    start_date = format_dt(start_date)
    end_date = format_dt(end_date)

    # Running query and parsing response
    response = requests.get(f'http://drax-production.herokuapp.com/api/1/{stream}?date_from={start_date}&date_to={end_date}&group_by={time_group}')
    r_json = response.json()

    return r_json

# Cell
def dict_col_to_cols(df:pd.DataFrame, value_col='value'):
    """Checks the `value_col`, if it contains dictionaries these are transformed into new columns which then replace it"""

    ## Checks the value col is found in the dataframe
    if value_col not in df.columns:
        return df

    if isinstance(df.loc[0, value_col], dict):
        df_values = pd.DataFrame(df[value_col].to_dict()).T
        df[df_values.columns] = df_values
        df = df.drop(columns=[value_col])

    return df

# Cell
def clean_nested_dict_cols(df):
    """Unpacks columns contining nested dictionaries"""
    # Calculating columns that are still dictionaries
    s_types = df.iloc[0].apply(lambda val: type(val))
    cols_with_dicts = s_types[s_types == dict].index

    while len(cols_with_dicts) > 0:
        for col_with_dicts in cols_with_dicts:
            # Extracting dataframes from dictionary columns
            df = dict_col_to_cols(df, col_with_dicts)

            # Recalculating columns that are still dictionaries
            s_types = df.iloc[0].apply(lambda val: type(val))
            cols_with_dicts = s_types[s_types == dict].index

    return df

# Cell
def set_dt_idx(df:pd.DataFrame, idx_name='local_datetime'):
    """
    Converts the start datetime to UK local time, then sets it as the index and removes the original datetime columns
    """

    idx_dt = pd.DatetimeIndex(pd.to_datetime(df['start'], utc=True)).tz_convert('Europe/London')
    idx_dt.name = idx_name

    df.index = idx_dt
    df = df.drop(columns=['start', 'end'])

    return df

def create_df_dt_rng(start_date, end_date, freq='30T', tz='Europe/London', dt_str_template='%Y-%m-%d'):
    """
    Creates a dataframe mapping between local datetimes and electricity market dates/settlement periods
    """

    # Creating localised datetime index
    s_dt_rng = pd.date_range(start_date, end_date, freq=freq, tz=tz)
    s_dt_SP_count = pd.Series(0, index=s_dt_rng).resample('D').count()

    # Creating SP column
    SPs = []
    for num_SPs in list(s_dt_SP_count):
        SPs += list(range(1, num_SPs+1))

    # Creating datetime dataframe
    df_dt_rng = pd.DataFrame(index=s_dt_rng)
    df_dt_rng.index.name = 'local_datetime'

    # Adding query call cols
    df_dt_rng['SP'] = SPs
    df_dt_rng['date'] = df_dt_rng.index.strftime(dt_str_template)

    return df_dt_rng

def clean_df_dts(df):
    """Cleans the datetime index of the passed DataFrame"""
    df = set_dt_idx(df)
    df = df[~df.index.duplicated()]

    df_dt_rng = create_df_dt_rng(df.index.min(), df.index.max())
    df = df.reindex(df_dt_rng.index)

    df['SP'] = df_dt_rng['SP'] # Adding settlement period designation

    return df

# Cell
def retrieve_stream_df(start_date:str, end_date:str, stream:str, time_group='30m', renaming_dict={}):
    """
    Makes the call to Electric Insights and parses the response into a dataframe which is returned

    Parameters:
        start_date: Start date for data given as a string in the form '%Y-%m-%d'
        end_date: End date for data given as a string in the form '%Y-%m-%d'
        stream: One of 'prices_ahead', 'prices_ahead', 'prices', 'temperatures' or 'emissions'
        time_group: One of '30m', '1h', '1d' or '7d'. The default is '30m'
        renaming_dict: Mapping from old to new column names
    """

    # Calling data and parsing into dataframe
    r_json = query_API(start_date, end_date, stream, time_group)
    df = pd.DataFrame.from_dict(r_json)

    # Handling entrys which are dictionarys
    df = clean_nested_dict_cols(df)

    # Setting index as localised datetime, reindexing with all intervals and adding SP
    df = clean_df_dts(df)

    # Renaming value col
    if 'value' in df.columns:
        df = df.rename(columns={'value':stream})

    if 'referenceOnly' in df.columns:
        df = df.drop(columns=['referenceOnly'])

    df = df.rename(columns=renaming_dict)

    return df

# Cell
def check_streams(streams='*'):
    """
    Checks that the streams given are a list containing only possible streams, or is all streams - '*'.
    """

    possible_streams = ['prices_ahead', 'prices', 'temperatures', 'emissions', 'generation-mix']

    if isinstance(streams, list):
        unrecognised_streams = list(set(streams) - set(possible_streams))

        if len(unrecognised_streams) == 0:
            return streams
        else:
            unrecognised_streams_to_print = ''.join(["'"+stream+"', " for stream in unrecognised_streams])[:-2]
            raise ValueError(f"Streams {unrecognised_streams_to_print} could not be recognised, must be one of: {', '.join(possible_streams)}")

    elif streams=='*':
        return possible_streams

    else:
        raise ValueError(f"Streams could not be recognised, must be one of: {', '.join(possible_streams)}")

# Cell
def retrieve_streams_df(start_date:str, end_date:str, streams='*', time_group='30m', renaming_dict={}):
    """
    Makes the calls to Electric Insights for the given streams and parses the responses into a dataframe which is returned

    Parameters:
        start_date: Start date for data given as a string in the form '%Y-%m-%d'
        end_date: End date for data given as a string in the form '%Y-%m-%d'
        streams: Contains 'prices_ahead', 'prices_ahead', 'prices', 'temperatures' or 'emissions', or is given as all, '*'
        time_group: One of '30m', '1h', '1d' or '7d'. The default is '30m'
    """

    df = pd.DataFrame()
    streams = check_streams(streams)

    for stream in streams:
        df_stream = retrieve_stream_df(start_date, end_date, stream, renaming_dict=renaming_dict)
        df[df_stream.columns] = df_stream

    return df

# Cell
def get_EI_data(
    start_date,
    end_date,
    streams='*',
    batch_freq='3M',
    renaming_dict={
        'pumpedStorage' : 'pumped_storage',
        'northernIreland' : 'northern_ireland',
        'windOnshore': 'wind_onshore',
        'windOffshore': 'wind_offshore',
        'prices_ahead' : 'day_ahead_price',
        'prices' : 'imbalance_price',
        'temperatures' : 'temperature',
        'totalInGperkWh' : 'gCO2_per_kWh',
        'totalInTperh' : 'TCO2_per_h'
    }
):
    # Preparing batch dates
    if (pd.to_datetime(end_date) - pd.to_datetime(start_date)) > pd.Timedelta(10, 'w'):
        *batch_start_dates, post_batch_start_date = pd.date_range(start_date, end_date, freq=f'{batch_freq}S').strftime('%Y-%m-%d')
        pre_batch_end_date, *batch_end_dates = (pd.date_range(start_date, end_date, freq=batch_freq)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        batch_start_dates, batch_end_dates = [], []
        pre_batch_end_date, post_batch_start_date = end_date, end_date

    batch_date_pairs = list(zip(batch_start_dates, batch_end_dates))

    if start_date != pre_batch_end_date:
        batch_date_pairs = [(start_date, pre_batch_end_date)] + batch_date_pairs

    if end_date != post_batch_start_date:
        end_date = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        batch_date_pairs = batch_date_pairs + [(post_batch_start_date, end_date)]

    # Retrieving data
    df = pd.DataFrame()

    for batch_start_date, batch_end_date in tqdm(batch_date_pairs):
        df_batch = retrieve_streams_df(batch_start_date, batch_end_date, streams, renaming_dict=renaming_dict)
        df = df.append(df_batch)

    return df

# Cell
get_EI_files = lambda data_dir: [f for f in os.listdir(data_dir) if 'csv' in f]

def bulk_retrieval(start_year=2009, end_year=2020, data_dir='data'):
    """Retrieves and saves in batches of years
    """
    EI_files = get_EI_files(data_dir)

    for year in range(start_year, end_year+1):
        if f'electric_insights_{year}.csv' not in EI_files:
            start_date, end_date = f'{year}-01-01 00:00', f'{year}-12-31 23:30'
            df_EI = get_EI_data(start_date, end_date)
            df_EI.to_csv(f'{data_dir}/electric_insights_{year}.csv')

    return

# Cell
def check_for_gappy_data(data_dir):
    EI_files = get_EI_files(data_dir)

    for EI_file in tqdm(EI_files):
        year = int(EI_file.split('_')[-1].split('.')[0])

        df_EI_year = pd.read_csv(f'{data_dir}/{EI_file}')
        current_ts = pd.Timestamp.now()

        df_EI_year = df_EI_year.set_index('local_datetime')
        df_EI_year.index = pd.to_datetime(df_EI_year.index, utc=True).tz_convert('Europe/London')

        if year < current_ts.year:
            missing_dates = list(set(pd.date_range(f'{year}-01-01 00:00', f'{year}-12-31 23:30', freq='30T', tz='Europe/London')) - set(df_EI_year.index))
        else:
            missing_dates = list(set(pd.date_range(f'{year}-01-01 00:00', df_EI_year.index.max().tz_convert(None), freq='30T', tz='Europe/London')) - set(df_EI_year.index))

        if len(missing_dates) > 0:
            warn(f'There are {len(missing_dates)} missing dates in the {year} dataframe')

# Cell
def retrieve_latest_data(data_dir='data'):
    EI_files = get_EI_files(data_dir)
    EI_years_downloaded = [int(f.split('_')[-1].split('.')[0]) for f in EI_files]

    current_ts = pd.Timestamp.now(tz='Europe/London')

    if current_ts.year not in EI_years_downloaded:
        start_date, end_date = f'{current_ts.year}-01-01 00:00', current_ts.strftime('%Y-%m-%d %H:%M')
        df_EI = get_EI_data(start_date, end_date)
        df_EI.to_csv(f'{data_dir}/electric_insights_{current_ts.year}.csv')

    else:
        df_EI = pd.read_csv(f'{data_dir}/electric_insights_{current_ts.year}.csv')

        df_EI = df_EI.set_index('local_datetime')
        df_EI.index = pd.to_datetime(df_EI.index, utc=True).tz_convert('Europe/London')
        dt_rng = pd.date_range(df_EI.index.max(), current_ts, freq='30T', tz='Europe/London')

        if dt_rng.size > 1:
            start_date = dt_rng[0] - pd.Timedelta(days=7)
            end_date = dt_rng[-1]

            try:
                df_EI_latest = get_EI_data(start_date, end_date)
                df_EI_trimmed = df_EI.drop(list(set(df_EI_latest.index) - (set(df_EI_latest.index) - set(df_EI.index))))
                df_EI_combined = df_EI_trimmed.append(df_EI_latest)
                df_EI_combined.to_csv(f'{data_dir}/electric_insights_{current_ts.year}.csv')
            except:
                warn(f'Could not retrieve any new data between {start_date} and {end_date}')

# Cell
def year_to_resource(year=2009):
    resource = {
      "name": f"electric-insights-{year}",
      "path": f"electric_insights_{year}.csv",
      "profile": "tabular-data-resource",
      "schema": {
        "fields": [
          {
            "name": "local_datetime",
            "type": "datetime",
            "format": "default",
            "title": "Local Datetime",
            "description": "Datetime index in the `Europe/London` timezone"
          },
          {
            "name": "day_ahead_price",
            "type": "number",
            "format": "default",
            "title": "Day Ahead Price",
            "description": "Price of electricity on the day-ahead market exchanges"
          },
          {
            "name": "SP",
            "type": "integer",
            "format": "default",
            "title": "Settlement Period",
            "description": "Half hour settlement period. Normally 1-48 apart from during clock changes when there will be 46 or 50 settlement periods."
          },
          {
            "name": "imbalance_price",
            "type": "number",
            "format": "default",
            "title": "Balancing Market Price",
            "description": "Price of electricity on the balancing market"
          },
          {
            "name": "valueSum",
            "type": "number",
            "format": "default",
            "title": "Value Sum",
            "description": "Unknown"
          },
          {
            "name": "temperature",
            "type": "number",
            "format": "default",
            "title": "Temperature",
            "description": "The temperature at noon averaged over the whole country"
          },
          {
            "name": "TCO2_per_h",
            "type": "integer",
            "format": "default",
            "title": "Tonnes CO2 per Hour",
            "description": "Tonnes of CO2 released each hour"
          },
          {
            "name": "gCO2_per_kWh",
            "type": "integer",
            "format": "default",
            "description": "Carbon intensity on the GB power grid",
            "title": "Grams CO2 per KiloWatt Hour"
          },
          {
            "name": "nuclear",
            "type": "number",
            "format": "default",
            "title": "Nuclear Output (GW)",
            "description": "Power output from nuclear plants in GB"
          },
          {
            "name": "biomass",
            "type": "number",
            "format": "default",
            "title": "Biomass Output (GW)",
            "description": "Power output from biomass plants in GB"
          },
          {
            "name": "coal",
            "type": "number",
            "format": "default",
            "title": "Coal Output (GW)",
            "description": "Power output from coal plants in GB"
          },
          {
            "name": "gas",
            "type": "number",
            "format": "default",
            "title": "Gas Output (GW)",
            "description": "Power output from gas plants in GB"
          },
          {
            "name": "hydro",
            "type": "number",
            "format": "default",
            "title": "Hydro Output (GW)",
            "description": "Power output from hydro plants in GB"
          },
          {
            "name": "wind",
            "type": "number",
            "format": "default",
            "title": "Wind Output (GW)",
            "description": "Power output from wind plants in GB"
          },
          {
            "name": "solar",
            "type": "number",
            "format": "default",
            "title": "Solar Output (GW)",
            "description": "Power output from solar plants in GB"
          },
          {
            "name": "demand",
            "type": "number",
            "format": "default",
            "title": "Demand (GW)",
            "description": "Total demand for power in GB"
          },
          {
            "name": "pumped_storage",
            "type": "number",
            "format": "default",
            "title": "Pumped Storage (GW)",
            "description": "Power output from Pumped Storage plants in GB"
          },
          {
            "name": "wind_onshore",
            "type": "number",
            "format": "default",
            "title": "Onshore Wind Output (GW)",
            "description": "Power output from onshore wind plants in GB"
          },
          {
            "name": "wind_offshore",
            "type": "number",
            "format": "default",
            "title": "Offshore Wind Output (GW)",
            "description": "Power output from offshore wind plants in GB"
          },
          {
            "name": "belgian",
            "type": "number",
            "format": "default",
            "title": "Belgian Interconnector (GW)",
            "description": "Power flow in the Belgian Interconnector"
          },
          {
            "name": "dutch",
            "type": "number",
            "format": "default",
            "title": "Dutch Interconnector (GW)",
            "description": "Power flow in the Dutch Interconnector"
          },
          {
            "name": "french",
            "type": "number",
            "format": "default",
            "title": "French Interconnector (GW)",
            "description": "Power flow in the French Interconnector"
          },
          {
            "name": "ireland",
            "type": "number",
            "format": "default",
            "title": "Ireland Interconnector (GW)",
            "description": "Power flow in the Ireland Interconnector"
          },
          {
            "name": "northern_ireland",
            "type": "number",
            "format": "default",
            "title": "Northern Ireland Interconnector (GW)",
            "description": "Power flow in the Northern Ireladn Interconnector"
          },
          {
            "name": "irish",
            "type": "number",
            "format": "default",
            "title": "Irish Interconnector (GW)",
            "description": "Net flow of the two Irish interconnectors"
          }
        ]
      }
    }

    return resource

# Cell
def construct_datapackage(start_year=2009, end_year=2021):
    datapackage = {
      "profile": "tabular-data-package",
      "resources": [year_to_resource(year) for year in range(start_year, end_year+1)],
      "keywords": [
        "electric insights"
      ],
      "contributors": [
        {
          "title": "Drax",
          "role": "data-source"
        },
        {
          "title": "Bourn, Ayrton",
          "role": "maintainer"
        }
      ],
      "name": "electric-insights",
      "title": "Electric Insights",
      "homepage": "https://github.com/AyrtonB/Electric-Insights"
    }

    return datapackage

# Cell
app = typer.Typer()

# Cell
@app.command()
def download(data_dir='data', start_year=2009, end_year=None):
    if end_year is None:
        end_year = pd.Timestamp.now().year

    bulk_retrieval(start_year=start_year, end_year=end_year-1, data_dir=data_dir)
    check_for_gappy_data(data_dir)
    retrieve_latest_data(data_dir)

    datapackage = construct_datapackage(start_year=start_year, end_year=end_year)

    with open(f'{data_dir}/datapackage.json', 'w') as f:
        json.dump(datapackage, f)

    return

# Cell
if __name__ == '__main__' and '__file__' in globals():
    app()