#!/usr/bin/env python
# -*- coding: utf-8 -*

"""Preprocess shapefiles to map ars to gemeindeschluessel."""

import os

import pyproj
import geopandas as gpd

import gzip
import pickle

__raw_data_path = os.path.join("data", "raw_data")
__preprocessed_data_path = os.path.join("data", "intermediate_data", 
                                        "electric_vehicles")

def map_ars_to_gemeinde(save_it: bool = True) -> dict:
    """Use the shapefiles provided by the 'Bundesamt für Kartographie und Geodäsie'
    to map the ARS to the Gemeindeschluessel."""

    gdf = gpd.read_file(__raw_data_path + 
                        '/geobund/vg250_ebenen_1231/VG250_VWG.shp').to_crs(epsg=4326)
    gdf.rename_geometry('WKT', inplace=True)

    dict_ars_gemeindeverbund = {ars_value: gemeindeverbund_key 
                                for ars_value, gemeindeverbund_key 
                                in zip(gdf.ARS, gdf.GEN)}
    
    if save_it:
        fpath_out = __preprocessed_data_path + '/ars_to_gemeindeverbund.pklz'
        with gzip.open(fpath_out, 'wb') as fh_out:
            pickle.dump(dict_ars_gemeindeverbund, fh_out)

    return gdf, dict_ars_gemeindeverbund
