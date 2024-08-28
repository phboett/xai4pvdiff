#!/usr/bin/env python
# -*- coding: utf-8 -*

"""Merge the previously preprocessed data into one file that will
be used in the ML model."""

import os
import sys

import numpy as np
import pandas as pd

from xai_diffusion_of_innovations.preprocessing.electric_vehicles \
    import preprocess_inkar, preprocess_charging, preprocess_kba


sys.path.append("code")
from utils.utils import col_id_ma, col_name_ma

__intermediate_data_path = os.path.join("data", "intermediate_data", 
                                        "electric_vehicles")
__input_for_ml_path = os.path.join("data", "input")


def preprocess_all_data(verbose: bool = True) -> None:
    """Preprocess all data for electric vehicles."""


    ## Preprocess data
    # KBA
    _ = preprocess_kba.preprocess_kba(save_data=True, verbose=verbose)

    # INKAR
    _ = preprocess_inkar.preprocess_inkar(save_data=True, verbose=verbose)


    # Charging stations
    _ = preprocess_charging.preprocess_charging_stations(save_data=True, verbose=verbose)

    return


def merge_all_preprocessed_data(charging_year=2023, verbose: bool = True,
                                save_it: bool = True) -> pd.DataFrame:
    """Merge all preprocessed data into one dataframe."""

    ## Load preprocessed data
    df_inkar = pd.read_pickle(os.path.join(__intermediate_data_path, 
                                           "inkar_ars_adjusted.pklz"), compression="gzip")
    df_kba = pd.read_pickle(os.path.join(__intermediate_data_path, 
                                         "kba_ars_adjusted.pklz"), compression="gzip")
    df_charging_stations = pd.read_pickle(os.path.join(__intermediate_data_path,
                                                        "charging_stations_per_year_ars_adjusted.pklz"), 
                                                        compression="gzip")    

    if verbose:
        print("Length of INKAR data:", len(df_inkar))
        print("Length of KBA data:", len(df_kba))
        print("Length of charging stations data:", len(df_charging_stations))
    
    ## Merge data
    # Start by merging the data from INKAR and KBA
    df_complete = df_inkar.merge(df_kba, on=col_id_ma, how='inner',
                                 validate='one_to_one')


    # Merge the data from charging stations
    key_year = 'chargingstations_before_' + str(charging_year)
    df_charging_cut = df_charging_stations[[col_id_ma, key_year]].copy()

    
    df_complete = df_complete.merge(df_charging_cut, on=col_id_ma, how='outer')
    
    # Fill nan with zeros for the charging stations
    df_complete[key_year] = df_complete[key_year].fillna(0)

    # Drop global radiation
    df_complete.drop(columns=['global radiation'], inplace=True)

    # norm vehicles with total vehicles
    df_complete['priv. Elektro (BEV) per vehicle'] = (df_complete['priv. Elektro (BEV)'].astype(float) / 
                                                      df_complete['priv. gesamt'].astype(float))
    df_complete.drop(columns=['priv. Elektro (BEV)', 'priv. gesamt'], inplace=True)

    if save_it:
        fpath_out = __input_for_ml_path + "/bev_input.csv"
        df_complete.to_csv(fpath_out, sep=";", index=False)

    return df_complete