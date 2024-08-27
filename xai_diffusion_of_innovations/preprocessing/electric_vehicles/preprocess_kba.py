#!/usr/bin/env python
# -*- coding: utf-8 -*

"""Preprocess the data from the KBA dataset to get
the amount of electric vehicles per ARS. 
This requires mapping the ARS (e.g., id of a Gemeindeverbund) to
a common ars with the other datasets to subsequently merge the data."""

import os
import sys

import numpy as np
import pandas as pd

import gzip
import pickle

from xai_diffusion_of_innovations.preprocessing.electric_vehicles.mapping_functions import map_to_common_ars

sys.path.append("code")
from utils.utils import col_id_ma, col_name_ma


__raw_data_path = os.path.join("data", "raw_data")
__intermediate_data_path = os.path.join("data", "intermediate_data", 
                                        "electric_vehicles")

def preprocess_kba(save_data: bool = True,
                   verbose: bool = False) -> pd.DataFrame:
    """Take the xls sheet of KBA where the Ars has been added.

    Args:
        save_data (bool, optional): _description_. Defaults to False.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """

    # Load data where the ARS has been added
    df_kba = pd.read_excel(os.path.join(__raw_data_path, 'kba',
                                        'fz27_202404_with_ars.xlsx'), 
                                        header=0, 
                           sheet_name='processed', dtype={'ars_land': str, 
                                                          'ars_rb': str, 
                                                          'ars_kreis':str, 
                                                          'ars_vb': str, 
                                                          'ars_gemeinde': str, 
                                                          'plz':str}) 
    
    df_kba.replace(to_replace = ['-','.'], 
                   value = [0.0, float('NaN')], 
                   inplace = True)
    
    # TODO explicitly set dtypes
    # Generate ARS values
    ars_col = (df_kba['ars_land'] + df_kba['ars_rb'] + 
               df_kba['ars_kreis'] + df_kba['ars_vb'])

    assert ars_col.apply(len).unique() == 9, "ARS must have length 9."

    df_kba.drop(columns=['ars_land', 'ars_rb', 'ars_kreis', 
                         'ars_vb', 'ars_gemeinde'], inplace=True)

    df_kba.insert(0, col_id_ma, ars_col)
    cols_to_keep = [col_id_ma, "priv. gesamt", "priv. Elektro (BEV)"]
    df_kba = df_kba[cols_to_keep]

    # Use Geo Data to map ARS to GemV 
    # TODO How did you create this list? A few are missing from the shapefile
    """
    fpath_ars_to_gemeindeverbund = os.path.join(__intermediate_data_path, 
                                                "ars_to_gemeindeverbund.pklz")
    if not os.path.exists(fpath_ars_to_gemeindeverbund):
        raise FileExistsError("The file 'ars_to_gemeindeverbund.plkz' does not exist. " +
                              "Run the script 'preprocess_geo.py' first.")
    else:
        with gzip.open(fpath_ars_to_gemeindeverbund, 'rb') as fh_in:
            dict_ars_gemeindeverbund = pickle.load(fh_in)
    """
    df_kba = df_kba.groupby(col_id_ma).agg({'priv. gesamt': 'sum',
                                            'priv. Elektro (BEV)': 'sum'}).reset_index()
    
    # Load mapping table
    df_verbund = pd.read_excel(os.path.join(__intermediate_data_path, 
                                            "ars_gemeindeverbund.xlsx"), 
                               converters={'ars': str})
    dict_ars_gemeindeverbund = {ars_value: gemeindeverbund_value for ars_value, 
                            gemeindeverbund_value in zip(df_verbund.ars, 
                                                         df_verbund.Gemeindeverbund)}

    
    df_kba.insert(1, col_name_ma, df_kba[col_id_ma].map(dict_ars_gemeindeverbund))

    # name Jahnatal manually
    #df_kba.at['145220275', col_name_ma] = 'Jahnatal'

    # Map to common ARS
    df_kba = map_to_common_ars(df_kba, "kba_mod", verbose=verbose)

    # Aggregate
    df_agg = df_kba.groupby(col_id_ma).agg({'priv. gesamt': 'sum',
                                            'priv. Elektro (BEV)': 'sum'}).reset_index()
    
    if save_data:
        df_agg.to_pickle(os.path.join(__intermediate_data_path, 
                                      "kba_ars_adjusted.pklz"),
                         compression="gzip")

    return df_agg
