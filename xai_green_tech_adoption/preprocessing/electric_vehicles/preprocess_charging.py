#!/usr/bin/env python
# -*- coding: utf-8 -*

"""Preprocess the data collecting charging 
stations for electric vehicles."""

import os
import sys

from tqdm import tqdm

import numpy as np
import pandas as pd

import geopandas as gpd
from shapely.geometry import Point

import datetime as dt

from xai_green_tech_adoption.preprocessing.electric_vehicles.mapping_functions import map_to_common_ars

from xai_green_tech_adoption.utils.utils import col_id_ma, col_name_ma


__raw_data_path = os.path.join("data", "raw_data")
__intermediate_data_path = os.path.join("data", "intermediate_data", 
                                        "electric_vehicles")


def create_GeoDataFrame_from_charging_stations() -> gpd.GeoDataFrame:

    ## Charging stations
    df_bnetz = pd.read_excel(os.path.join(__raw_data_path, 
                                          'bnetza', 
                                          'BNetzA_chargingstations.xls'), 
                             header=10, converters={'Postleitzahl': str})
    col_to_keep = ['Betreiber', 'Straße', 'Hausnummer', 'Adresszusatz', 'Postleitzahl',
                   'Ort', 'Bundesland', 'Kreis/kreisfreie Stadt', 'Breitengrad',
                   'Längengrad', 'Inbetriebnahmedatum']
    
    df_bnetz = df_bnetz[col_to_keep]

    # track number of charging stations which will be merged later
    df_bnetz['chargingstations'] = 1
    df_bnetz.rename(columns={'Postleitzahl': 'plz'}, inplace=True)

    df_bnetz['Inbetriebnahme'] = pd.to_datetime(df_bnetz['Inbetriebnahmedatum'],
                                                format='%Y-%m-%d %H:%M:%S')
    
    # Modify geodata
    df_bnetz = df_bnetz.astype({'Breitengrad':'string'})
    df_bnetz['Breitengrad'] = df_bnetz['Breitengrad'].str.replace(',','.')
    df_bnetz = df_bnetz.astype({'Längengrad':'string'})
    df_bnetz['Längengrad'] = df_bnetz['Längengrad'].str.replace(',','.')
    df_bnetz = df_bnetz.astype({'Breitengrad':'float64','Längengrad':'float64'})

    df_bnetz['point'] = [Point(lon, lat) for lon, lat in zip(df_bnetz['Längengrad'], 
                                                             df_bnetz['Breitengrad'])]

    gdf_bnetz = gpd.GeoDataFrame(df_bnetz, geometry='point')
    gdf_bnetz.rename_geometry('WKT', inplace=True)

    return gdf_bnetz


def map_charging_stations_to_ars(gdf_bnetz: gpd.GeoDataFrame,
                                 gdf_bund: gpd.GeoDataFrame,
                                 include_autobahn: bool, 
                                 verbose: bool = False) -> gpd.GeoDataFrame:

    if include_autobahn:
        gdf_bnetz_internal = gdf_bnetz.copy()
    else:
        mask_no_autobahn = (~gdf_bnetz.Straße.str.contains('Autobahn') & 
                            ~gdf_bnetz.Straße.str.contains('Raststätte'))
        gdf_bnetz_internal = gdf_bnetz[mask_no_autobahn].copy().reset_index(drop=True)

        if verbose:
            print(f"Removed {np.count_nonzero(~mask_no_autobahn)} " + 
                  "charging stations located on the 'Autobahn'.")

    # Initialize the column for the ARS
    gdf_bnetz_internal[col_id_ma] = ''

    points_not_mapped = list()

    for idx, row in tqdm(gdf_bnetz_internal.iterrows(), 
                         total=len(gdf_bnetz_internal)):
        
        point = row.WKT
        
        in_polygon_mask = point.within(gdf_bund.geometry)

        if np.count_nonzero(in_polygon_mask) == 0:
            points_not_mapped.append(point)
            continue

        elif np.count_nonzero(in_polygon_mask) > 1:
            raise ValueError(f"Charging station {row.WKT} is in multiple polygons.")

        ars_r = gdf_bund.ARS[in_polygon_mask].values[0]
        
        gdf_bnetz_internal.iloc[idx, 14] = ars_r

    #gdf_bnetz_internal.ars = gdf_bnetz_internal.ars.explode()
    gdf_bnetz_internal[col_id_ma] = gdf_bnetz_internal[col_id_ma].astype(str)

    assert all((gdf_bnetz_internal[col_id_ma].apply(len) == 9) |\
               (gdf_bnetz_internal[col_id_ma].apply(len) == 0)), "ARS must have length 9."
    
    return gdf_bnetz_internal, points_not_mapped


def aggregate_charging_stations_different_years(gdf_mapped: gpd.GeoDataFrame,
                                                year_range: tuple[int, int]=(1992, 2023)) -> None:
    
    year_range = np.arange(min(year_range), max(year_range)+1)

    #list_aggregated = list()

    # Shouldn't all appear in the end again? i.e. every in gdf_mapped should be back again

    # initialize Dataframe
    df_bnetz_grouped = pd.DataFrame({col_id_ma:[]}).set_index(col_id_ma)

    for year_r in tqdm(year_range, desc='Aggreg. per year'):
        time_threshold = dt.datetime(year_r, 1, 1)

        mask_time = gdf_mapped['Inbetriebnahme'] <= time_threshold
        gdf_mapped_year = gdf_mapped[mask_time][[col_id_ma, 'chargingstations']]

        gdf_mapped_year_sum = gdf_mapped_year.groupby(col_id_ma).agg({'chargingstations': 'sum'})
        gdf_mapped_year_sum.rename({'chargingstations': f'chargingstations_before_{year_r}'}, 
                                   axis=1, inplace=True)

        #list_aggregated.append(gdf_mapped_year_sum)
        
        df_bnetz_grouped = df_bnetz_grouped.merge(gdf_mapped_year_sum, 
                                                  how = 'outer',
                                                  left_index=True, right_index=True)

    #df_bnetz_grouped = pd.concat([list_aggregated]).groupby(col_id_ma).agg('sum')

    return df_bnetz_grouped


def preprocess_charging_stations(include_autobahn: bool = False, 
                                 save_data: bool = True,
                                 verbose: bool = False) -> None:


    ## Load data (shape_files)
    # Shapefile of Federal states                         
    gdf_bund = gpd.read_file(os.path.join(__raw_data_path, 'geobund', 
                                     'vg250_ebenen_1231', 
                                     'VG250_VWG.shp')).to_crs(epsg=4326)
    gdf_bund.rename_geometry('WKT', inplace=True)

    gdf_bnetz = create_GeoDataFrame_from_charging_stations()
    len_bnetz_t0 = len(gdf_bnetz)

    gdf_mapped, points_not_mapped = map_charging_stations_to_ars(gdf_bnetz, gdf_bund, 
                                                                 include_autobahn,
                                                                 verbose=verbose)

    # Some could not be mapped automatically and were mapped by hand
    df_manual_ars = pd.read_excel(os.path.join(__raw_data_path,'bnetza', 
                                               'manual_changes_bnetza.xls'),
                                    dtype = {'ARS_land':str, 'ARS_rb':str, 
                                           'ARS_kreis':str, 'ARS_gv':str, 
                                           'plz':'str'})
    df_manual_ars[col_id_ma] = (df_manual_ars['ARS_land'] + df_manual_ars['ARS_rb'] + 
                                df_manual_ars['ARS_kreis'] + df_manual_ars['ARS_gv'])

    df_manual_ars = df_manual_ars.drop(columns = ['ARS_land', 
                                                  'ARS_rb', 'ARS_kreis', 'ARS_gv'])
    df_manual_ars.rename(columns={'point':'WKT'}, inplace=True)

    # Remove non mapped (i.e. with str of zero) and add manually mapped
    mask_gdf_not_mapped = gdf_mapped[col_id_ma].apply(len) == 0
    gdf_mapped.drop(gdf_mapped.index[mask_gdf_not_mapped], inplace=True)

    gdf_mapped = pd.concat([gdf_mapped, df_manual_ars]).reset_index(drop=True)

    if verbose:
        len_bnetz_final = len(gdf_mapped)
        print(f"\nStations before vs after: {len_bnetz_t0}" + 
              f" vs {len_bnetz_final} (diff: {len_bnetz_t0 - len_bnetz_final})\n")

    df_charging_stations_per_year = aggregate_charging_stations_different_years(gdf_mapped)

    # Add Gemeindename from file
    df_verbund = pd.read_excel(os.path.join(__raw_data_path, 
                                            "ars_to_gemeindeverbund.xls"), 
                               converters={'ars': str})
    dict_ars_gemeindeverbund = {ars_value: gemeindeverbund_value for ars_value, 
                            gemeindeverbund_value in zip(df_verbund.ars, 
                                                         df_verbund.Gemeindeverbund)}
    
    df_charging_stations_per_year[col_name_ma] = df_charging_stations_per_year.index.map(dict_ars_gemeindeverbund)
    df_charging_stations_per_year.reset_index(inplace=True)

    # Adjust to common ars across data sets
    df_ars_adjusted = map_to_common_ars(df_charging_stations_per_year, 
                                        "charging_mod", verbose=verbose)

    df_ars_adjusted = df_ars_adjusted.groupby(col_id_ma).agg('sum').reset_index()

    if save_data:
        fpath_out = __intermediate_data_path + '/charging_stations_per_year_ars_adjusted'
        if include_autobahn:
            fpath_out += '_with_autobahn'
        df_ars_adjusted.to_pickle(fpath_out + ".pklz", compression="gzip")

    return df_ars_adjusted
