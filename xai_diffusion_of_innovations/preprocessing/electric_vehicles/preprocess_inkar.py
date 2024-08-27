#!/usr/bin/env python
# -*- coding: utf-8 -*

"""Preprocess the data from the INKAR dataset.
This requires mapping the ARS (e.g., id of a Gemeindeverbund)
to that of a different date to subsequently merge the data."""

import os
import sys

import numpy as np
import pandas as pd

from xai_diffusion_of_innovations.preprocessing.electric_vehicles.mapping_functions import map_to_common_ars

sys.path.append("code")
from utils.utils import col_id_ma, col_name_ma,name_of_aggregated_sum_features

__input_data_path = os.path.join("data", "input")
__output_data_path = os.path.join("data", "intermediate_data", 
                                  "electric_vehicles")

# Create output directory if it does not exist
if not os.path.exists(__output_data_path):
    os.makedirs(__output_data_path)

def preprocess_inkar(save_data: bool = True, 
                     verbose: bool = False) -> pd.DataFrame:
    """_summary_

    Args:
        save_data (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """

    # Load preprocessed data from inkar
    df = pd.read_csv(os.path.join(__input_data_path, "input.csv"), sep=";",
                     dtype={col_id_ma: str})

    # pad to get length of 9
    df[col_id_ma] = df[col_id_ma].str.pad(9, fillchar="0")

    # TODO What do with 'accessability to {regional, middle-order} centre'? Drop it?
    df_mod = map_to_common_ars(df, "inkar_mod", verbose=verbose)

    # Aggregate data of fused regions either by sum or by weighted average (population)
    # Wether to use weighted average or sum is defined in the list name_of_aggregated_sum_features.
    weight_fct = lambda xx: np.average(xx, weights=df_mod.loc[xx.index, "population"])
    
    cols_to_agg = [xx for xx in df_mod.columns if xx not in [col_id_ma, col_name_ma]]
    dict_aggregate = {xx: 'sum' if xx in name_of_aggregated_sum_features else weight_fct for xx in cols_to_agg}
    df_agg = df_mod.groupby([col_id_ma, col_name_ma]).agg(dict_aggregate).reset_index()

    if save_data:
        df_agg.to_pickle(os.path.join(__output_data_path, 
                                      "inkar_ars_adjusted.pklz"),
                         compression="gzip")
    
    return df_agg