#!/usr/bin/env python
# -*- coding: utf-8 -*

"""Merge the previously preprocessed data into one file that will
be used in the ML model."""

import os
import sys

import numpy as np
import pandas as pd

from xai_diffusion_of_innovations.preprocessing.electric_vehicles.mapping_functions import get_similarity_two_str


sys.path.append("code")
from utils.utils import col_id_ma, col_name_ma

__intermediate_data_path = os.path.join("data", "intermediate_data", 
                                        "electric_vehicles")
__input_for_ml_path = os.path.join("data", "input")


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
    # Fill nan with zeros
    df_complete[key_year] = df_complete[key_year].fillna(0)

    if save_it:
        df_complete.to_hdf(os.path.join(__intermediate_data_path, 
                                        "complete_data.h5"), key="df", mode="w")

    return df_complete