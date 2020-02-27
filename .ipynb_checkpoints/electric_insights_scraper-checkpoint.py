# -*- coding: utf-8 -*-
"""
Data scraper for Electric Insights
Dates are given in UK local time, generation in GW
Author: Ayrton Bourn - ayrtonbourn@outlook.com
"""


""" Imports """
import requests
import pandas as pd

import warnings
from datetime import date

import matplotlib.pyplot as plt
import matplotlib.transforms as mtf


""" Caller """
class Caller(object):
    def query_API(self, start_date:str, end_date:str, stream:str, time_group='30m'):
        """
        'Query API' makes the call to Electric Insights and parses the JSON response

        Arguments:
            * start_date - Start date for data given as a string in the form '%Y-%m-%d'
            * end_date - End date for data given as a string in the form '%Y-%m-%d'
            * stream - One of 'prices_ahead', 'prices_ahead', 'prices', 'temperatures' or 'emissions'
            * time_group - One of '30m', '1h', '1d' or '7d'. The default is '30m'
        """

        ## Checking stream is an EI endpoint
        assert stream in self.possible_streams, f"Stream must be one of {''.join([stream+', ' for stream in self.possible_streams])[:-2]}"

        ## Checking time_group will be accepted by API
        assert time_group in self.time_groups, f"Time group must be one of {''.join([time_group+', ' for time_group in self.time_groups])[:-2]}"

        ## Formatting dates
        format_dt = lambda dt: date.strftime(dt, '%Y-%m-%d') if isinstance(dt, date) else dt
        start_date = format_dt(start_date)
        end_date = format_dt(end_date)

        ## Running query and parsing response
        response = requests.get(f'http://drax-production.herokuapp.com/api/1/{stream}?date_from={start_date}&date_to={end_date}&group_by={time_group}')
        r_json = response.json()

        return r_json

    def dict_col_2_cols(self, df:pd.DataFrame, value_col='value'):
        """
        Checks the value col, if it contains dictionarys these are transformed into new columns which then replace it
        """

        ## Checks the value col is found in the dataframe
        if value_col not in df.columns:
            return df

        if isinstance(df.loc[0, value_col], dict):
            df_values = pd.DataFrame(df[value_col].to_dict()).T
            df[df_values.columns] = df_values
            df = df.drop(columns=[value_col])

        return df

    def set_dt_idx(self, df:pd.DataFrame, idx_name='local_datetime'):
        """
        Converts the start datetime to UK local time, then sets it as the index and removes the original datetime columns
        """

        idx_dt = pd.DatetimeIndex(pd.to_datetime(df['start'], utc=True)).tz_convert('Europe/London')
        idx_dt.name = idx_name

        df.index = idx_dt
        df = df.drop(columns=['start', 'end'])

        return df
    
    def create_df_dt_rng(self, start_date, end_date, freq='30T', tz='Europe/London', dt_str_template='%Y-%m-%d'):
        ## Creating localised datetime index
        s_dt_rng = pd.date_range(start_date, end_date, freq=freq, tz=tz)
        s_dt_SP_count = pd.Series(0, index=s_dt_rng).resample('D').count()

        ## Creating SP column
        SPs = []
        for num_SPs in list(s_dt_SP_count):
            SPs += list(range(1, num_SPs+1))

        ## Creating datetime dataframe
        df_dt_rng = pd.DataFrame(index=s_dt_rng)
        df_dt_rng.index.name = 'local_datetime'

        ## Adding query call cols
        df_dt_rng['SP'] = SPs
        df_dt_rng['date'] = df_dt_rng.index.strftime(dt_str_template)

        return df_dt_rng

    def call_stream(self, start_date:str, end_date:str, stream:str, time_group='30m'):
        """
        'Call Stream' makes the call to Electric Insights and parses the response into a dataframe which is returned

        Arguments:
            * start_date - Start date for data given as a string in the form '%Y-%m-%d'
            * end_date - End date for data given as a string in the form '%Y-%m-%d'
            * stream - One of 'prices_ahead', 'prices_ahead', 'prices', 'temperatures' or 'emissions'
            * time_group - One of '30m', '1h', '1d' or '7d'. The default is '30m'
        """

        ## Calling data and parsing into dataframe
        r_json = self.query_API(start_date, end_date, stream, time_group)
        df = pd.DataFrame.from_dict(r_json)

        ## Handling entrys which are dictionarys
        df = self.dict_col_2_cols(df)
        
        s_types = df.iloc[0].apply(lambda val: type(val))
        cols_with_dicts = s_types[s_types == dict].index
        
        if len(cols_with_dicts) > 0:
            for col_with_dicts in cols_with_dicts:
                df = self.dict_col_2_cols(df, col_with_dicts)
            
        ## Setting index as localised datetime, reindexing with all intervals and adding SP
        df = self.set_dt_idx(df)
        df = df[~df.index.duplicated()] ## Dropping duplicates 
        
        df_dt_rng = self.create_df_dt_rng(df.index.min(), df.index.max())
        df = df.reindex(df_dt_rng.index)
        
        df['SP'] = df_dt_rng['SP'] # Adding settlement period designation

        ## Renaming value col
        if 'value' in df.columns:
            df = df.rename(columns={'value':stream})
            
        if 'referenceOnly' in df.columns:
            df = df.drop(columns=['referenceOnly'])
            
        df = df.rename(columns=self.renaming_dict)

        return df
    
    def check_streams(self, streams):
        """
        Checks that the streams given are a list containing only possible streams, or is all streams, '*'.
        """
        
        if isinstance(streams, list):
            unrecognised_streams = list(set(self.possible_streams) - set(streams))
            
            if len(unrecognised_streams) == 0:
                return streams 
            else:
                unrecognised_streams_2_print = ''.join(["'"+stream+"', " for stream in unrecognised_streams])[:-2]
                raise ValueError(f'Streams {unrecognised_streams_2_print} could not be recognised')
            
        elif streams=='*':
            return self.possible_streams 
            
        else:
            raise ValueError('Streams could not be recognised')

    def call_streams(self, start_date:str, end_date:str, streams='*', time_group='30m'):
        """
        'Call Streams' makes the calls to Electric Insights for the given streams and parses the responses into a dataframe which is returned

        Arguments:
            * start_date - Start date for data given as a string in the form '%Y-%m-%d'
            * end_date - End date for data given as a string in the form '%Y-%m-%d'
            * streams - Contains 'prices_ahead', 'prices_ahead', 'prices', 'temperatures' or 'emissions', or is given as all, '*'
            * time_group - One of '30m', '1h', '1d' or '7d'. The default is '30m'
        """

        df = pd.DataFrame()
        streams = self.check_streams(streams)
            
        for stream in streams:
            df_stream = self.call_stream(start_date, end_date, stream)           
            df[df_stream.columns] = df_stream

        return df
    
    def __init__(self):
        self.possible_streams = ['prices_ahead', 'prices', 'temperatures', 'emissions', 'generation-mix']
        self.time_groups = ['30m', '1h', '1d', '7d']
        self.renaming_dict = {
            'prices_ahead' : 'day_ahead_price',
            'prices' : 'imbalance_price',
            'temperatures' : 'temperature',
            'totalInGperkWh' : 'gCO2_per_kWh',
            'totalInTperh' : 'TCO2_per_h',
            'pumpedStorage' : 'pumped_storage',
            'northernIreland' : 'northern_ireland'
        }
        
        
        
##################
## Plotter Help ##
##################

def rgb_2_plt_tuple(rgb_tuple):
    plt_tuple = tuple([x/255 for x in rgb_tuple])
    return plt_tuple

fuel_color_dict = {
    'Imports & Storage' : rgb_2_plt_tuple((121,68,149)), 
    'nuclear' : rgb_2_plt_tuple((77,157,87)), 
    'biomass' : rgb_2_plt_tuple((168,125,81)), 
    'gas' : rgb_2_plt_tuple((254,156,66)), 
    'coal' : rgb_2_plt_tuple((122,122,122)), 
    'hydro' : rgb_2_plt_tuple((50,120,196)), 
    'wind' : rgb_2_plt_tuple((72,194,227)), 
    'solar' : rgb_2_plt_tuple((255,219,65)),
              }

fuel_color_list = list(fuel_color_dict.values())

fuel_order = ['Imports & Storage', 'nuclear', 'biomass', 'gas', 'coal', 'hydro', 'wind', 'solar']
interconnectors = ['french', 'irish', 'dutch', 'belgian', 'ireland', 'northern_ireland']

def clean_df(df, freq='7D'):
    df = (df
          .copy()
          .assign(imports_storage=df[interconnectors+['pumped_storage']].sum(axis=1))
          .rename(columns={'imports_storage':'Imports & Storage'})
          .drop(columns=interconnectors+['demand', 'pumped_storage'])
          [fuel_order]
         )

    df_resampled = df.astype('float').resample(freq).mean()
    return df_resampled

## Plotting 
def quick_plot(df, ax=None, save_path=None, dpi=150):
    if ax == None:
        fig = plt.figure(figsize=(10, 5), dpi=dpi)
        ax = plt.subplot()

    ax.stackplot(df.index.values, df.values.T, labels=df.columns.str.capitalize(), linewidth=0.25, edgecolor='white', colors=fuel_color_list)

    plt.rcParams['axes.ymargin'] = 0
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlim(df.index.min(), df.index.max())
    ax.legend(ncol=4, bbox_to_anchor=(0.85, 1.15), frameon=False)
    ax.set_ylabel('Generation (GW)')

    if save_path:
        fig.savefig(save_path)
        
    return ax