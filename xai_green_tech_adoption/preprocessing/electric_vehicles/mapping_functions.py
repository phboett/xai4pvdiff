#!/usr/bin/env python
# -*- coding: utf-8 -*


"""Functions used to map data with a ARS to a different ARS, as well as, 
aggregating data of fused regions."""

import os
import numpy as np
import pandas as pd

import re
from difflib import SequenceMatcher
from itertools import product

from xai_green_tech_adoption.utils.utils import col_id_ma, col_name_ma

__raw_data_path = os.path.join("data", "raw_data")



def get_similarity_two_str(str1: str, str2: str, 
                           verbose: bool = True) -> float:
    """Get the similarity between two strings. This is used
    to compare GemV names. It outputs a number between 0 and 1 
    that indicates the similarity between the two strings."""
    
    if not isinstance(str1, str) or not isinstance(str2, str):
        return np.nan

    str1 = str1.replace("/ ", "/")
    str2 = str2.replace("/ ", "/")

    str1 = str1.replace(". ", ".")
    str2 = str2.replace(". ", ".")

    replacements = ["(VGem)", "VVG der Stadt ", "VVG der Gemeinde ", 
                    "EG ", "Amt ", "Kirchspielslandgemeinden ", 
                    "Kirchspielslandgemeinde ", "GVV "]

    for replacement in replacements:
        str1 = str1.replace(replacement, "")
        str2 = str2.replace(replacement, "")

    if ", " in str1:
        str1_comp = str1.split(", ")[0]
    else:
        str1_comp = str1

    if ", " in str2:
        str2_comp = str2.split(", ")[0]

    else:
        str2_comp = str2

    # Remove things in parentheses
    str1_comp = re.sub(r'\([^)]*\)', '', str1_comp)
    str2_comp = re.sub(r'\([^)]*\)', '', str2_comp)

    similarity = SequenceMatcher(None, str1_comp.strip(), 
                           str2_comp.strip()).ratio()

    return similarity


def similarity_of_substrings(str1: str, str2: str, min_str_len: int = 3, 
                             verbose: bool = True) -> float:

    # Prepare the string by replacing all signs by whitespace 
    # and subsequently stripping the whitespaces
    replace_signs = ["-", ",", "/", ".", "(VGem)", "VVG der Stadt", 
                     "VVG der Gemeinde", "Kurort", 
                     "EG ", "Amt ", "Kirchspielslandgemeinden", 
                     "Kirchspielslandgemeinde", "GVV"]
    
    for replace_r in replace_signs:
        str1 = str1.replace(replace_r, ' ')
        str2 = str2.replace(replace_r, ' ')

    str_ls1 = str1.split()
    str_ls2 = str2.split()

    uni_combi_ls = list(list(zip(str_ls1, ele)) for ele in product(str_ls2, 
                                                    repeat=len(str_ls2)))
    flattened_ls = [xx for xs in uni_combi_ls for xx in xs]
    score_ls = [SequenceMatcher(None, xx[0], xx[1]).ratio() 
                for xx in flattened_ls if len(xx[0]) > min_str_len 
                and len(xx[1])> min_str_len]
    max_score = max(score_ls)
    
    if verbose:
        max_idx = np.argmax(score_ls)
        print(flattened_ls[max_idx], end=', with a score of ')
        print(score_ls[max_idx])

    return max_score


def map_to_common_ars(df_in: pd.DataFrame, sheet_name: str,
                      verbose: bool = False) -> pd.DataFrame:
    """Map the ARS to a different ARS using a mapping table that
    was created manually."""

    df_internal = df_in.copy()

    fpath_in = os.path.join(__raw_data_path , 
                            "bev_manual_changes.xls")
    
    # Drop the ones that are not needed according to the mapping table
    df_drop = pd.read_excel(fpath_in, sheet_name="discard", 
                            dtype=str, header=0)
    assert all(df_drop.ars_ref.apply(len) == 9), "String of ARS must have length 9."

    list_to_drop = df_drop.ars_ref.to_list()
    mask_drop = df_internal[col_id_ma].isin(list_to_drop)

    df_internal.drop(df_internal.index[mask_drop], inplace=True)
    nr_rows_dropped = np.count_nonzero(mask_drop)
    if verbose:
        print(f"Dropped {nr_rows_dropped} rows.")

    # Map the ARS to the one of a different date
    df_changes = pd.read_excel(fpath_in, sheet_name=sheet_name, dtype=str, header=0)
    assert all(df_changes.ars_ref.apply(len) == 9) and \
           all(df_changes.ars_mod.apply(len) == 9), \
           "String of ARS must have length 9."
    
    df_unchanged = df_internal.copy()

    for idx_r, row_r in df_changes.iterrows():
        ars_ref = row_r.ars_ref
        ars_mod = row_r.ars_mod

        name_ref = row_r.gemeinde_ref
        name_mod = row_r.gemeinde_mod
        try:
            # Find index of element that will be changed
            index_r = df_internal.index[df_unchanged[col_id_ma] == ars_ref]
            index_r_val = index_r.values[0]
        
            if len(index_r) == 1:
                sim_score = get_similarity_two_str(df_internal.at[index_r_val, col_name_ma], name_ref)
                
                assert  sim_score > 0.9, \
                    f"Original name (manual list): {name_ref} || Actual name: {df_internal.at[index_r_val, col_name_ma]} " +\
                    f"at ARS: {ars_ref}"
                
                if verbose:
                    print(f"ARS {ars_ref} ({name_ref}) changed to {ars_mod} ({name_mod}).")

                df_internal.at[index_r_val, col_id_ma] = ars_mod
                df_internal.at[index_r_val, col_name_ma] = name_mod
            
            else: 
                raise ValueError(f"There are multiple entries for ARS: {ars_ref}.")
        except IndexError:
            print(f"ARS {ars_ref} not found in the data.")
            raise IndexError

    return df_internal

