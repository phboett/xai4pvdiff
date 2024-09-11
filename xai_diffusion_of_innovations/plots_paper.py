#!/usr/bin/env python
# -*- coding: utf-8 -*

"""Plot for producing the figures for the paper 
that presents analysis of the diffusion of both pv and bev."""

import os
import sys

import numpy as np
import pandas as pd

import geopandas as gpd

from matplotlib import pyplot as plt
from PIL import Image

plt.rcParams.update({'text.usetex': True})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

import shap

import pickle
import gzip

__fpath_output = "data/output"
__cartoon_path = "data/raw_data/plotting"

sys.path.append("code")
from utils.utils import col_feature_count, ranking_mean_r2_desc, \
    col_run_id, mean_r2_cv_test, mean_r2_cv_train, col_file_path, \
    col_bev_per_vehicle, col_power_accum_pv, col_features, \
    col_idx_train, col_idx_val, col_idx_test, mean_r2_cv_test, col_mean_shap,\
    col_id_ma, col_name_ma, features_norm_to_population_ls, features_norm_drop_ls

sys.path.append("code/shap_analysis")
import plotting
from supporting_functions import prepare_performance_dataframe, \
    prepare_metadata_dataframe, get_fitted_models as __get_fitted_models, \
    get_mean_shap as __get_mean_shap

if not os.path.exists("plots/"):
    os.makedirs("plots/")


rename_tick_dict = {'employees with academic qualification': 'employees with academic\n qualification',
                        'completed buildings with renewable heat energy systems': 'completed buildings with\n renewable heat energy systems',
                        'completed flats with renewable heat energy systems': 'completed flats with\n renewable heat energy systems',
                        'completed (semi-) detached houses (per capita)': 'completed (semi-) detached\n houses (per capita)',
                        'per capita permissions for (semi-) detached houses': 'per capita permissions\n for (semi-) detached houses',
                        'density of residents and employees': 'density of residents\n and employees',
                        'employees with academic qualification': 'employees with\n academic qualification',
                        'GVA per employee in secondary sector': 'GVA per employee\n in secondary sector',
                        'German university entrance qualification (Abitur)': 'Germany university entrance\n qualification (Abitur)',
                        'employees in knowledge-intensive industries': 'employees in\n knowledge-intensive industries',
                        'median income (professional qualification)': 'mean income\n (professional qualification)',
                        'Certificates of Secondary Education (males)': 'certificates of secondary\neducation (males)',
                        'employees without any professional qualification': 'employees without any\nprofessional qualification',
                        'CDU/CSU': 'votes CDU/CSU [\%]',
                        'other parties': 'votes other parties [\%]',
                        'FDP': 'votes FDP [\%]',
                        'The Left': 'votes The Left [\%]',
                        'AfD': 'votes AfD [\%]',
                        'The Greens': 'votes The Greens [\%]',
                        'SPD': 'votes SPD [\%]',
                        'share 4-person households': 'share of\n4-person households [\%]',
                        'share 5-person households': 'share of\n5-person households [\%]',
                        'flats with 5+ rooms': 'share of\nflats with 5+ rooms [\%]',
                        col_power_accum_pv: 'photovoltaic power [kW p.h.]',
                        'global radiation': 'global radiation [kWh/m²]',
                        'income tax': 'income tax [€ p.c.]',}

def load_performance_and_metadata_dataframes(use_normalized: bool = False):

    norm_str = ""
    if use_normalized:
        norm_str = "_norm"

    df_perf_pv = pd.read_csv(f"{__fpath_output}/results_rfe{norm_str}.csv", 
                             sep=';')
    df_perf_pv = prepare_performance_dataframe(df_perf_pv, 'features')

    df_perf_bev = pd.read_csv(f"{__fpath_output}/results_rfe_bev{norm_str}.csv", 
                              sep=';')
    df_perf_bev = prepare_performance_dataframe(df_perf_bev, 'features')

    
    df_metadata_pv = pd.read_csv(f"{__fpath_output}/metadata_rfe{norm_str}.csv",
                                    sep=';')
    df_metadata_pv = prepare_metadata_dataframe(df_metadata_pv, 
                                                ["indices_training_set", 
                  "indices_val_set", 
                  "indices_test_set"])  

    df_metadata_bev = pd.read_csv(f"{__fpath_output}/metadata_rfe_bev{norm_str}.csv",
                                    sep=';')
    df_metadata_bev = prepare_metadata_dataframe(df_metadata_bev, 
                                                    ["indices_training_set", 
                  "indices_val_set", 
                  "indices_test_set"]) 
    
    return df_perf_pv, df_perf_bev, df_metadata_pv, df_metadata_bev


def get_fitted_models_and_mean_shap_values(df_perf, df_metadata,
                                           feature_count_threshold,
                                           target_feature):

    model_dict = __get_fitted_models(
        df_perf=df_perf,
        df_metadata=df_metadata,
        feat_count_red_model=feature_count_threshold,
        target_feat=target_feature,
        train_on_train_val=True, show_progress=True)
    
    df_mean_shap = __get_mean_shap(
        model_dict,
        feature_count_threshold=feature_count_threshold,
        df_perf=df_perf,
        df_metadata=df_metadata)

    return model_dict, df_mean_shap


def get_details_about_best_model(df_perf: pd.DataFrame, df_metadata: pd.DataFrame, 
                                 model_dict: dict,
                                 split_id_best_model: int,
                                 feature_count_threshold: int,
                                 use_normalized: bool = False):
    
    df_data = pd.read_csv(df_metadata[col_file_path].unique()[0], sep=";")
    if use_normalized:
        norm_ls = features_norm_to_population_ls
        drop_ls = features_norm_drop_ls
        df_data = df_data.drop(columns=drop_ls)

        for feat_r in norm_ls:
            feat_new = feat_r + '_per_capita'
            df_data[feat_new] = df_data[feat_r] / df_data['population'].astype(float)
            df_data.drop(columns=feat_r, 
                        inplace=True)

    # Get reduced model and the input data for the reduced model
    df_performance_red_model = df_perf.loc[
        (df_perf[col_run_id] == split_id_best_model)
        & (df_perf[col_feature_count] == feature_count_threshold)
        & (df_perf[ranking_mean_r2_desc] == 1),
        :,
    ]

    X_red_model = df_data[df_performance_red_model[col_features].values[0]]
    red_model = model_dict[split_id_best_model]

    # Compute SHAP and SHAP interaction values for the reduced model
    tree_explainer = shap.explainers.Tree(red_model)
    shap_values = tree_explainer.shap_values(X_red_model)
    interaction_values = tree_explainer.shap_interaction_values(X_red_model)


    return [X_red_model, red_model, 
            shap_values, interaction_values]


def plot_map_distribution_pv_and_bev(save_fig: bool = False, 
                                     map_type: str = "GEM",
                                     cmap='inferno',
                                     label_size=16):
    """Plot the distribution of PV and BEV on a map.

    Args:
        save_fig (bool, optional): _description_. Defaults to False.
        map_type (str, optional): Type of shapefile used, which can be 
            "GEM" for Gemeinden or "VWG" for Gemeindeverbünde. Defaults to "GEM".

    Returns:
        _type_: _description_
    """

    # Load dataset
    data_dtype_dict = {col_id_ma: int}
    df_data_pv = pd.read_csv("data/input/input.csv", sep=';', 
                             dtype=data_dtype_dict)
    #df_data_pv[col_id_ma] = df_data_pv[col_id_ma].str.pad(9, fillchar="0")
    df_data_bev = pd.read_csv("data/input/bev_input.csv", sep=';', 
                              dtype=data_dtype_dict)
    #df_data_bev[col_id_ma] = df_data_bev[col_id_ma].str.pad(9, fillchar="0")
    

    # Load dictionary connecting ars to wkt coordinates
    if map_type == "GEM":
        # TODO str or int for ARS and AGS... one not both
        gdf_bund = gpd.read_file("data/maps/VG250_GEM.shp", 
                                 decimal=',', encoding='UTF-8',
                                 dtype={'AGS': int, 'ARS': int})
        gdf_bund.AGS = gdf_bund.AGS.astype(int)
        gdf_bund.ARS = gdf_bund.ARS.astype(int)

        df_mapping = pd.read_csv("data/intermediate_data/mapping_municipalities_2000_2019.csv",
                    sep=";",
                    dtype={"Official municipality code (AGS)": int,
                           col_id_ma: int})
        df_mapping.rename(columns={"Official municipality code (AGS)": "AGS"}, inplace=True)
        
        df_bund_n_mapping = pd.merge(left=gdf_bund, right=df_mapping[['AGS', col_name_ma, col_id_ma]],
                                     on='AGS', how='left')
        
        gdf_complete_pv = gpd.GeoDataFrame(pd.merge(left=df_bund_n_mapping, 
                                right=df_data_pv, on=col_id_ma, how='left'), 
                                geometry='geometry')
        
        gdf_complete_bev = gpd.GeoDataFrame(pd.merge(left=df_bund_n_mapping, 
                                right=df_data_bev, on=col_id_ma, how='left'), 
                                geometry='geometry')
        
    
    elif map_type == "VWG":

        raise NotImplementedError("VWG not implemented yet.")
        gdf_bund = gpd.read_file("data/raw_data/geobund/" + 
                             "vg250_ebenen_1231/VG250_VWG.shp", 
                                dtype=data_dtype_dict)
        #df_bund.ARS = df_bund.ARS.apply(lambda x: x[:9])
        #df_bund.ARS = df_bund.ARS.str.pad(9, fillchar="0")

        #assert all(df_bund.ARS.apply(len) == 9), "Not all ARS are of length 9."

        dict_ars_wkt = {row_r['ARS']: row_r['geometry'] for idx_r, row_r in 
                        gdf_bund.iterrows()}


        df_data_pv['geometry'] = df_data_pv[col_id_ma].map(dict_ars_wkt)
        df_data_bev['geometry'] = df_data_bev[col_id_ma].map(dict_ars_wkt)
        
        return df_data_pv, df_data_bev


    # Plot
    fig, [ax_pv, ax_bev] = plt.subplots(1, 2, figsize=(14, 10),
                                        sharex='all', sharey='all')

    gdf_complete_pv.plot(column=col_power_accum_pv, ax=ax_pv,
                         cmap=cmap)
    gdf_complete_bev.plot(column=col_bev_per_vehicle, ax=ax_bev,
                          cmap=cmap)

    # Aesthetics
    fig.set_dpi(300)
    for ax_r in [ax_pv, ax_bev]:
        ax_r.axis('off')
        ax_r.set_rasterized(True)

    #fig.tight_layout()
    fig.subplots_adjust(right=0.9, bottom=0.15)
    
    # Colorbar
    pos_ax_pv = ax_pv.get_position()
    pos_ax_bev = ax_bev.get_position()

    shorten = .1
    cax = fig.add_axes([pos_ax_pv.x0 + shorten, pos_ax_pv.y0 - 0.05, 
                           pos_ax_bev.x1 - pos_ax_pv.x0 - 2*shorten, 0.02])
    
    min_target_value = df_data_pv[col_power_accum_pv].values.min()
    max_target_value = df_data_pv[col_power_accum_pv].values.max()
    sm = plt.cm.ScalarMappable(cmap=cmap, 
                               norm=plt.Normalize(vmin=min_target_value, 
                                                  vmax=max_target_value))
    
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', 
                        ticks=[min_target_value, max_target_value])

    cbar.ax.set_xticklabels(["minimal value of\ntarget feature", 
                             "maximal value of\ntag feature"],)
    cbar.ax.tick_params(labelsize=label_size)
    cbar.solids.set(alpha=1.)

    # Add cartoons
    pv_image = Image.open(f"{__cartoon_path}/pv.png")
    ax_pv_image = fig.add_axes([0.35, .7, 0.17, 0.17])
    ax_pv_image.imshow(pv_image)
    ax_pv_image.axis('off')


    car_image = Image.open(f"{__cartoon_path}/car.png")
    ax_car_image = fig.add_axes([0.825, .7, 0.15, 0.15])
    ax_car_image.imshow(car_image)
    ax_car_image.axis('off')

    if save_fig:
        fig_path = "plots/map_target_distribution_pv_and_bev.pdf"
        fig.savefig(fig_path, bbox_inches='tight')

        fig.clear()
        plt.close(fig)

    else:
        plt.show()

    return 


def plot_parts_method(save_fig: bool = False,
                      cmap='inferno', use_normalized: bool = True, 
                      feature_count_threshold_pv: int = 15):
    """Plot the parts of the figure explaining the approach"""

    # Load target data
    data_dtype_dict = {col_id_ma: int}
    df_data_pv = pd.read_csv("data/input/input.csv", sep=';', 
                             dtype=data_dtype_dict)
    #df_data_pv[col_id_ma] = df_data_pv[col_id_ma].str.pad(9, fillchar="0")
    
    # Target + Input Map
    # TODO str or int for ARS and AGS... one not both
    gdf_bund = gpd.read_file("data/maps/VG250_GEM.shp", 
                                decimal=',', encoding='UTF-8',
                                dtype={'AGS': int, 'ARS': int})
    gdf_bund.AGS = gdf_bund.AGS.astype(int)
    gdf_bund.ARS = gdf_bund.ARS.astype(int)

    df_mapping = pd.read_csv("data/intermediate_data/mapping_municipalities_2000_2019.csv",
                sep=";",
                dtype={"Official municipality code (AGS)": int,
                        col_id_ma: int})
    df_mapping.rename(columns={"Official municipality code (AGS)": "AGS"}, inplace=True)
    
    df_bund_n_mapping = pd.merge(left=gdf_bund, right=df_mapping[['AGS', col_name_ma, col_id_ma]],
                                    on='AGS', how='left')
    
    gdf_complete_pv = gpd.GeoDataFrame(pd.merge(left=df_bund_n_mapping, 
                            right=df_data_pv, on=col_id_ma, how='left'), 
                            geometry='geometry')
    
    
    fig_target = plt.figure(figsize=(10, 10))
    ax_target = fig_target.add_subplot(111)
    gdf_complete_pv.plot(column=col_power_accum_pv, ax=ax_target,
                         cmap=cmap)
    
    fig_input = plt.figure(figsize=(10, 10))
    ax_input = fig_input.add_subplot(111)
    gdf_complete_pv.plot(column='ownership occupation ratio', 
                         ax=ax_input,
                         cmap=cmap)
    
    # Shap Plots
    ## Example
    [df_perf_pv, df_meta_pv, 
     _, _] = load_performance_and_metadata_dataframes(use_normalized=use_normalized)

    df_predictions = pd.read_csv("data/output/predictions_rfe.csv", sep=";")
    
    df_perf_all_split_pv = df_perf_pv.loc[
    (df_perf_pv[col_feature_count] == feature_count_threshold_pv)
    & (df_perf_pv[ranking_mean_r2_desc] == 1),
    [col_run_id, mean_r2_cv_test],
    ]

    split_id_best_model = df_perf_all_split_pv.loc[
        df_perf_all_split_pv[mean_r2_cv_test] == df_perf_all_split_pv[mean_r2_cv_test].max(),
        col_run_id,
    ].values[0]
    print("The id of train-test split with best performance for PV: ", 
          split_id_best_model)
    
    model_dict_pv, df_mean_shap_pv = get_fitted_models_and_mean_shap_values(df_perf_pv, df_meta_pv,
                                                                            feature_count_threshold_pv, col_power_accum_pv)
    # TODO is this the df_mean_shap the same? It is, right?
    [X_red_model_pv, shap_values_pv, _, interaction_values_pv,
     _, _, _, _] = get_reduced_model_features_n_shap(feature_count_threshold_pv, 
                                                     feature_count_threshold_bev=15, 
                                                     run_evaluation=False,
                                                     use_normalized=use_normalized)
    
    red_model = model_dict_pv[split_id_best_model]
    y_red = red_model.predict(X_red_model_pv)
    tree_explainer = shap.explainers.Tree(red_model)    
    shap_values = tree_explainer.shap_values(X_red_model_pv)

    # Plot for a specific city (picked a random example)
    idx_city = df_data_pv.loc[df_data_pv[col_id_ma] == 53340020].index
    shap_city = shap_values[idx_city, :]
    shap_city = shap_city[0]
    idx_sorted = np.argsort(np.abs(shap_city))[::-1]

    fig_shap_addition = plt.figure(figsize=(10, 10))
    ax_shap_addition = fig_shap_addition.add_subplot(111)

    nn_features_shown = 3
    for idx, ranking_feature in enumerate(range(nn_features_shown)):
        if shap_city[idx_sorted[ranking_feature]] >= 0:
            ax_shap_addition.broken_barh(
                [(mean_plus_shap, shap_city[idx_sorted[ranking_feature]])],
                (11 + idx * 10, 7),
                facecolors=("firebrick"),
                zorder=2,
            )
            mean_plus_shap += shap_city[idx_sorted[ranking_feature]]
        else:
            ax_shap_addition.broken_barh(
                [
                    (
                        (mean_plus_shap + shap_city[idx_sorted[ranking_feature]]),
                        np.abs(shap_city[idx_sorted[ranking_feature]]),
                    )
                ],
                (11 + idx * 10, 7),
                facecolors=("royalblue"),
                zorder=2,
            )
            mean_plus_shap += shap_city[idx_sorted[ranking_feature]]
    if nn_features_shown < feature_count_threshold_pv:
        shap_remaining = sum(shap_city[idx_sorted[k:]])
        if shap_remaining >= 0:
            ax_shap_addition.broken_barh(
                [(mean_plus_shap, shap_remaining)],
                (11 + nn_features_shown * 10, 7),
                facecolors=("firebrick"),
                zorder=2,
            )
        else:
            ax_shap_addition.broken_barh(
                [(mean_plus_shap + shap_remaining, np.abs(shap_remaining))],
                (11 + nn_features_shown * 10, 7),
                facecolors=("royalblue"),
                zorder=2,
            )
        ax_shap_addition.set_ylim(5, (nn_features_shown + 2) * 10)
        ax_shap_addition.set_yticks(
            range(15, 15 + (nn_features_shown + 1) * 10, 10),
            # labels=[
            #     "global radiation"
            #     if X_red_model.columns[ranking_feature] == "Global irradiation"
            #     else X_red_model.columns[ranking_feature]
            #     for ranking_feature in idx_sorted[:k]
            # ]
            labels=X_red_model_pv.columns[idx_sorted[:k]].to_list() + ["remaining features"],
        )

        mean_plus_shap += shap_remaining
    else:
        ax_shap_addition.set_ylim(5, (nn_features_shown + 1) * 10)
        ax_shap_addition.set_yticks(
            range(15, 15 + nn_features_shown * 10, 10),
            labels=[
                X_red_model_pv.columns[ranking_feature] for ranking_feature in idx_sorted[:k]
            ],
        )
    ax_shap_addition.invert_yaxis()
    
    plt.axvline(x=np.mean(y_pred), color="darkgray", zorder=1, linewidth=3)
    plt.axvline(x=mean_plus_shap, color="darkgray", ls="--", zorder=1, linewidth=3)
    ax_shap_addition.spines[["right", "bottom", "top", "left"]].set_visible(False)

    ax_shap_addition.tick_params(axis="both", which="both", fontsize=20)
    ax_shap_addition.tick_params(left=False)
    ax_shap_addition.tick_params(axis="x", which="both", 
                                 bottom=False, top=False, labelbottom=False)

    # Aesthetics
    ax_target.axis('off')
    ax_input.axis('off')

    if save_fig:
        fpath_fig_target = "plots/method_target_map.svg"
        fpath_fig_input = "plots/method_input_map.svg"
        fpath_shap_addtion = "plots/method_shap_addition.svg"
        fpath_feature_importance = "plots/method_feature_importance.svg"

        fig_target.savefig(fpath_fig_target, bbox_inches='tight')
        fig_input.savefig(fpath_fig_input, bbox_inches='tight')

        for fig_r in [fig_target, fig_input]:
            fig_r.clear()
            plt.close(fig_r)

    else:

        plt.show()


    return


def plot_recursive_gbt_performance(feature_count_threshold_pv: int = 15, 
                                   feature_count_threshold_bev: int = 15,
                                   save_fig: bool = False, 
                                   use_normalized: bool = True):

    [df_perf_pv, df_perf_bev, 
     _, _] = load_performance_and_metadata_dataframes(use_normalized=use_normalized)
    
    df_perf_all_split_pv = df_perf_pv.loc[
    (df_perf_pv[col_feature_count] == feature_count_threshold_pv)
    & (df_perf_pv[ranking_mean_r2_desc] == 1),
    [col_run_id, mean_r2_cv_test],
    ]

    split_id_best_model = df_perf_all_split_pv.loc[
        df_perf_all_split_pv[mean_r2_cv_test] == df_perf_all_split_pv[mean_r2_cv_test].max(),
        col_run_id,
    ].values[0]
    print("The id of train-test split with best performance for PV: ", 
          split_id_best_model)
    
    score_at_threshold_pv = np.mean(df_perf_all_split_pv[mean_r2_cv_test])
    print(f"The mean R2 score at threshold for PV: {score_at_threshold_pv:.5f}")
    df_perf_all_split_above_threshold_pv = df_perf_pv.loc[
    (df_perf_pv[col_feature_count] >= feature_count_threshold_pv)
    & (df_perf_pv[ranking_mean_r2_desc] == 1),
    [col_run_id, mean_r2_cv_test],
    ]
    mean_before_threshold_pv = np.mean(df_perf_all_split_above_threshold_pv[mean_r2_cv_test])
    std_before_threshold_pv = np.std(df_perf_all_split_above_threshold_pv[mean_r2_cv_test])
    print(f"Mean and std R2 score above threshold for " + 
          f"PV: {mean_before_threshold_pv:.5f}, {std_before_threshold_pv:.7f}")

    df_perf_all_split_bev = df_perf_bev.loc[
    (df_perf_bev[col_feature_count] == feature_count_threshold_bev)
    & (df_perf_bev[ranking_mean_r2_desc] == 1),
    [col_run_id, mean_r2_cv_test],
    ]
    split_id_best_model = df_perf_all_split_bev.loc[
        df_perf_all_split_bev[mean_r2_cv_test] == df_perf_all_split_bev[mean_r2_cv_test].max(),
        col_run_id,
    ].values[0]
    print("\n\nThe id of train-test split with best performance for BEV: ", 
          split_id_best_model)
    
    score_at_threshold_bev = np.mean(df_perf_all_split_bev[mean_r2_cv_test])
    print(f"The mean R2 score at threshold for bev: {score_at_threshold_bev:.5f}")
    df_perf_all_split_above_threshold_bev = df_perf_bev.loc[
    (df_perf_bev[col_feature_count] >= feature_count_threshold_bev)
    & (df_perf_bev[ranking_mean_r2_desc] == 1),
    [col_run_id, mean_r2_cv_test],
    ]
    mean_before_threshold_bev = np.mean(df_perf_all_split_above_threshold_bev[mean_r2_cv_test])
    std_before_threshold_bev = np.std(df_perf_all_split_above_threshold_bev[mean_r2_cv_test])
    print(f"Mean and std R2 score above threshold for bev" + 
          f": {mean_before_threshold_bev:.5f}, {std_before_threshold_bev:.7f}")

    # Plot
    fig, [ax_pv, ax_bev] = plt.subplots(1, 2, figsize=(12, 6), 
                                        sharey='all')

    ## Plot PV
    plotting.ax_performance(ax_pv, 
                df_perf_pv,
                perf_metric_test=mean_r2_cv_test,
                perf_metric_train=mean_r2_cv_train,
                list_runs=df_perf_pv[col_run_id].unique(),
                x_max=210,
                x_min=-5,
                s=110,
                run_red_model=split_id_best_model,
                feat_count_red_model=feature_count_threshold_pv,
                include_train_score=True,
                indicate_red_model=True,
                label_yaxis=True, show_legend=False)
    
    plotting.ax_performance(ax_bev, 
                df_perf_bev,
                perf_metric_test=mean_r2_cv_test,
                perf_metric_train=mean_r2_cv_train,
                list_runs=df_perf_bev[col_run_id].unique(),
                x_max=210,
                x_min=-5,
                s=110,
                run_red_model=split_id_best_model,
                feat_count_red_model=feature_count_threshold_bev,
                include_train_score=True,
                indicate_red_model=True,
                label_yaxis=False, show_legend=True)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.75)
    # Add cartoons
    pv_image = Image.open(f"{__cartoon_path}/pv.png")
    ax_pv_image = fig.add_axes([0.325, .75, 0.25, 0.25])
    ax_pv_image.imshow(pv_image)
    ax_pv_image.axis('off')


    car_image = Image.open(f"{__cartoon_path}/car.png")
    ax_car_image = fig.add_axes([0.8, .75, 0.175, 0.175])
    ax_car_image.imshow(car_image)
    ax_car_image.axis('off')

    

    if save_fig:
        fig_path = "plots/rfe_performance_pv_and_bev.pdf"
        fig.savefig(fig_path, bbox_inches='tight')

        fig.clear()
        plt.close(fig)

    else:
        plt.show()

    return


def plot_mean_shap_features(feature_count_threshold_pv: int = 15,
                            feature_count_threshold_bev: int = 15,  
                            save_fig: bool = False, 
                            run_evaluation: bool = False,
                            use_normalized: bool = True):
    

    [_, _, df_mean_shap_pv, _,
     _, _, df_mean_shap_bev, _] = get_reduced_model_features_n_shap(feature_count_threshold_pv, 
                                                                    feature_count_threshold_bev, 
                                                                    run_evaluation,
                                                                    use_normalized=use_normalized)
    feature_occ_lims = (1, 10)

    features_pv = list(df_mean_shap_pv.index).copy()
    features_pv.remove(mean_r2_cv_test)

    features_bev = list(df_mean_shap_bev.index).copy()
    features_bev.remove(mean_r2_cv_test)

    # Plot
    fig, [ax_pv, ax_bev] = plt.subplots(1, 2, figsize=(20, 17))
    plotting.bar_shap_feature_imp(df_mean_shap_pv, features_pv,
                                  ax=ax_pv, nr_features_shown=25,
                                  feature_occurence_lims=feature_occ_lims,
                                   cmap='viridis', labelsize=18)
    sm = plotting.bar_shap_feature_imp(df_mean_shap_bev, features_bev, 
                                  ax=ax_bev, nr_features_shown=25,
                                  feature_occurence_lims=feature_occ_lims,
                                  cmap='viridis', labelsize=18)
    
    fig.tight_layout()

    cax = fig.add_axes([.925, .5, 0.02 , .3])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('number feature occurs in runs', rotation=270, labelpad=25, size=18)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_ticks([1, 4, 7, 10])

    # Add cartoons
    pv_image = Image.open(f"{__cartoon_path}/pv.png")
    ax_pv_image = fig.add_axes([0.25, .1, 0.25, 0.25])
    ax_pv_image.imshow(pv_image)
    ax_pv_image.axis('off')


    car_image = Image.open(f"{__cartoon_path}/car.png")
    ax_car_image = fig.add_axes([0.775, .075, 0.2, 0.2])
    ax_car_image.imshow(car_image)
    ax_car_image.axis('off')

    if save_fig:
        fig_path = "plots/shap_feature_importance_pv_n_bev.pdf"
        fig.savefig(fig_path, bbox_inches='tight')

        fig.clear()
        plt.close(fig)

    else:
        plt.show()
    
    return


def get_reduced_model_features_n_shap(feature_count_threshold_pv, 
                                      feature_count_threshold_bev, 
                                      run_evaluation=False,
                                      use_normalized: bool = True):

    norm_str = ""
    if use_normalized:
        norm_str = "_norm"

    fpath_plot_data = (f"data/intermediate_data/" + 
                       f"dependency_plots_pv_and_bev{norm_str}" + 
                       f"_nrfeatPV{feature_count_threshold_pv}" + 
                       f"_nrfeatBEV{feature_count_threshold_bev}.pklz")

    if not os.path.exists(fpath_plot_data) or run_evaluation: 
        [df_perf_pv, df_perf_bev, 
        df_metadata_pv, df_metadata_bev] = load_performance_and_metadata_dataframes(use_normalized=use_normalized)
        
        ## PV
        df_perf_all_split_pv = df_perf_pv.loc[
        (df_perf_pv[col_feature_count] == feature_count_threshold_pv)
        & (df_perf_pv[ranking_mean_r2_desc] == 1),
        [col_run_id, mean_r2_cv_test],
        ]
        split_id_best_model = df_perf_all_split_pv.loc[
            df_perf_all_split_pv[mean_r2_cv_test] == df_perf_all_split_pv[mean_r2_cv_test].max(),
            col_run_id,
        ].values[0]
        print("The id of train-test split with best performance for PV: ", 
            split_id_best_model)

        model_dict_pv, df_mean_shap_pv = get_fitted_models_and_mean_shap_values(df_perf_pv, df_metadata_pv,
                                                                                feature_count_threshold_pv, col_power_accum_pv)

        [X_red_model_pv, _, 
        shap_values_pv, interaction_values_pv] = get_details_about_best_model(df_perf_pv, df_metadata_pv, model_dict_pv, 
                                                                                    split_id_best_model, feature_count_threshold_pv)

        
        ## BEV
        df_perf_all_split_bev = df_perf_bev.loc[
        (df_perf_bev[col_feature_count] == feature_count_threshold_bev)
        & (df_perf_bev[ranking_mean_r2_desc] == 1),
        [col_run_id, mean_r2_cv_test],
        ]
        split_id_best_model = df_perf_all_split_bev.loc[
            df_perf_all_split_bev[mean_r2_cv_test] == df_perf_all_split_bev[mean_r2_cv_test].max(),
            col_run_id,
        ].values[0]
        print("The id of train-test split with best performance for bev: ", 
            split_id_best_model)

        model_dict_bev, df_mean_shap_bev = get_fitted_models_and_mean_shap_values(df_perf_bev, df_metadata_bev,
                                                                                feature_count_threshold_bev, col_bev_per_vehicle)

        [X_red_model_bev, _, 
        shap_values_bev, interaction_values_bev] = get_details_about_best_model(df_perf_bev, df_metadata_bev, model_dict_bev, 
                                                                                    split_id_best_model, feature_count_threshold_bev)

        res_tup = (X_red_model_pv, shap_values_pv, df_mean_shap_pv, interaction_values_pv,
                   X_red_model_bev, shap_values_bev, df_mean_shap_bev, interaction_values_bev)
        
        with gzip.open(fpath_plot_data, "wb") as fh_plot_data:
            pickle.dump(res_tup, fh_plot_data)
    
    else:
        with gzip.open(fpath_plot_data, "rb") as fh_plot_data:
            (X_red_model_pv, shap_values_pv, df_mean_shap_pv, interaction_values_pv,
             X_red_model_bev, shap_values_bev, df_mean_shap_bev, interaction_values_bev) = pickle.load(fh_plot_data)

    return [X_red_model_pv, shap_values_pv, df_mean_shap_pv, interaction_values_pv,
            X_red_model_bev, shap_values_bev, df_mean_shap_bev, interaction_values_bev]


def plot_dependency_pv_and_bev(feature_count_threshold_pv=15,
                              feature_count_threshold_bev=15, 
                              save_fig=False, nr_best_features=4,
                              run_evaluation=False, use_normalized=True):
    """Plot 2 by 4 with depency plot in the best model."""

    [X_red_model_pv, shap_values_pv, df_mean_shap_pv, _,
    X_red_model_bev, shap_values_bev, df_mean_shap_bev, _] = get_reduced_model_features_n_shap(feature_count_threshold_pv, 
                                                                                            feature_count_threshold_bev, 
                                                                                            run_evaluation,
                                                                                            use_normalized=use_normalized)

    # Plot
    fig, [ax_pv, ax_bev] = plt.subplots(2, nr_best_features, 
                                        figsize=(5*nr_best_features, 10),                    
                                        sharey='row')

    ## PV row
    ## Choose list of best features
    ranking_features_pv = df_mean_shap_pv.loc[:, 
                                              col_mean_shap].drop(mean_r2_cv_test, 
                                                                  axis=0)
    count_plotted = 0

    for feat_r in list(ranking_features_pv.index)[::-1]:
        if feat_r in X_red_model_pv.columns:
            xlabel_feat = rename_tick_dict[feat_r] if feat_r in rename_tick_dict else feat_r
            plotting.dependence_plot(X_red_model_pv, shap_values_pv,
                                     feature=feat_r,
                                     x_label=xlabel_feat,
                                     y_label="", label_size=20,
                                     ax=ax_pv[count_plotted])
            
            count_plotted += 1

        else:
            print(f"Feature {feat_r} not in reduced model.")

        if count_plotted == nr_best_features:
            break

    ## BEV row
    ranking_features_bev = df_mean_shap_bev.loc[:, 
                                              col_mean_shap].drop(mean_r2_cv_test, 
                                                                  axis=0)
    count_plotted = 0

    for feat_r in list(ranking_features_bev.index)[::-1]:
        if feat_r in X_red_model_bev.columns:
            xlabel_feat = rename_tick_dict[feat_r] if feat_r in rename_tick_dict else feat_r
            plotting.dependence_plot(X_red_model_bev, shap_values_bev,
                                     feature=feat_r,
                                     x_label=xlabel_feat,
                                     y_label="", label_size=20,
                                     ax=ax_bev[count_plotted])
            
            count_plotted += 1

        else:
            print(f"Feature {feat_r} not in reduced model.")

        if count_plotted == nr_best_features:
            break

    # Aesthetics

    for ax_r in [ax_pv[0], ax_bev[0]]:
        ax_r.set_ylabel("SHAP", size=20)  

    for ax_r in np.concatenate((ax_pv, ax_bev)):
        ax_r.axhline(y=0, color='black', linestyle='--', linewidth=1.5)

    fig.tight_layout()
    fig.subplots_adjust(right=0.87, hspace=.4)
    # Add cartoons
    pv_image = Image.open(f"{__cartoon_path}/pv.png")
    ax_pv_image = fig.add_axes([0.815, .75, 0.24, 0.24])
    ax_pv_image.imshow(pv_image)
    ax_pv_image.axis('off')


    car_image = Image.open(f"{__cartoon_path}/car.png")
    ax_car_image = fig.add_axes([0.875, .33, 0.122, 0.122])
    ax_car_image.imshow(car_image)
    ax_car_image.axis('off')

    
    if save_fig:
        fig_path = "plots/dependency_plots_pv_and_bev.pdf"
        fig.savefig(fig_path, bbox_inches='tight')

        fig.clear()
        plt.close(fig)

    else:
        plt.show()

    return


def plot_all_dependencies_separate(feature_count_threshold_pv=15,
                                   feature_count_threshold_bev=15,
                                   save_fig=False, run_evaluation=False,
                                   labelsize=16, use_normalized=True):
    """Plot shap values over the feature values for all features of the best number."""


    [X_red_model_pv, shap_values_pv, _, _,
    X_red_model_bev, shap_values_bev, _, _] = get_reduced_model_features_n_shap(feature_count_threshold_pv, 
                                                                                feature_count_threshold_bev, 
                                                                                run_evaluation,
                                                                                use_normalized=use_normalized)
    
    # Plot
    ## PV
    mean_shap_red_pv = np.mean(abs(shap_values_pv), axis=0)
    idx_mean_shap_sort_pv = np.argsort(mean_shap_red_pv)[::-1]
    fig_cols_pv = len(mean_shap_red_pv) // 5
    remainder_pv = len(mean_shap_red_pv) % 5
    if remainder_pv > 0:
        fig_cols_pv += 1
    fig_pv, ax_arr_pv = plt.subplots(fig_cols_pv, 5, 
                                 figsize=(20, 4 * fig_cols_pv),                    
                                 sharey='all')
    
    idx_plot = 0
    for idx_feat in idx_mean_shap_sort_pv:
        feat_r = X_red_model_pv.iloc[:, idx_feat].name
        xlabel_feat = rename_tick_dict[feat_r] if feat_r in rename_tick_dict else feat_r
        plotting.dependence_plot(X_red_model_pv, shap_values_pv,
                                feature=feat_r,
                                x_label=xlabel_feat,
                                y_label="", label_size=labelsize,
                                ax=ax_arr_pv.flatten()[idx_plot])
        ax_arr_pv.flatten()[idx_plot].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        idx_plot += 1
        
    if idx_plot < len(ax_arr_pv.flatten()):
        for idx_r in range(idx_plot, len(ax_arr_pv.flatten())):
            ax_arr_pv.flatten()[idx_r].axis('off')


    ## BEV
    mean_shap_red_bev = np.mean(abs(shap_values_bev), axis=0)
    idx_mean_shap_sort_bev = np.argsort(mean_shap_red_bev)[::-1]
    fig_cols_bev = len(mean_shap_red_bev) // 5
    remainder_bev = len(mean_shap_red_bev) % 5
    if remainder_bev > 0:
        fig_cols_bev += 1
    fig_bev, ax_arr_bev = plt.subplots(fig_cols_bev, 5, 
                                 figsize=(20, 5 * fig_cols_bev),                    
                                 sharey='all')
    
    idx_plot = 0
    for idx_feat in idx_mean_shap_sort_bev:
        feat_r = X_red_model_bev.iloc[:, idx_feat].name
        xlabel_feat = rename_tick_dict[feat_r] if feat_r in rename_tick_dict else feat_r
        plotting.dependence_plot(X_red_model_bev, shap_values_bev,
                                feature=feat_r,
                                x_label=xlabel_feat,
                                y_label="", label_size=labelsize,
                                ax=ax_arr_bev.flatten()[idx_plot])
        ax_arr_bev.flatten()[idx_plot].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        idx_plot += 1
        
    if idx_plot < len(ax_arr_bev.flatten()):
        for idx_r in range(idx_plot, len(ax_arr_bev.flatten())):
            ax_arr_bev.flatten()[idx_r].axis('off')

    # Aesthetics
    for ax_r in ax_arr_pv[:,0].flatten():
        ax_r.set_ylabel("SHAP", size=labelsize)

    for ax_r in ax_arr_bev[:,0].flatten():
        ax_r.set_ylabel("SHAP", size=labelsize)

    fig_pv.tight_layout()
    fig_pv.subplots_adjust(hspace=.5, right=0.925)
    fig_pv.canvas.manager.set_window_title("Feature depence plots for PV")

    fig_bev.tight_layout()
    fig_bev.subplots_adjust(hspace=.5, right=0.925)
    fig_bev.canvas.manager.set_window_title("Feature depence plots for BEV")

    # Add cartoons
    pv_image = Image.open(f"{__cartoon_path}/pv.png")
    ax_pv_image = fig_pv.add_axes([0.875, .825, 0.17, 0.17])
    ax_pv_image.imshow(pv_image)
    ax_pv_image.axis('off')


    car_image = Image.open(f"{__cartoon_path}/car.png")
    ax_car_image = fig_bev.add_axes([0.89, .89, 0.105, 0.105])
    ax_car_image.imshow(car_image)
    ax_car_image.axis('off')

    if save_fig:
        fig_path = "plots/all_dependency_best_model_pv.pdf"
        fig_pv.savefig(fig_path, bbox_inches='tight')

        fig_pv.clear()
        plt.close(fig_pv)

        fig_path = "plots/all_dependency_best_model_bev.pdf"
        fig_bev.savefig(fig_path, bbox_inches='tight')

        fig_bev.clear()
        plt.close(fig_bev)

    else:
        plt.show()

    return

def plot_interaction_heatmaps_pv_and_bev(feature_count_threshold_pv: int = 15, 
                                    feature_count_threshold_bev:int =15, 
                                    save_fig=False, run_evaluation=False, 
                                    use_normalized=True):


    # Load data

    [X_red_model_pv, shap_values_pv, df_mean_shap_pv, interaction_values_pv,
     X_red_model_bev, shap_values_bev, df_mean_shap_bev, interaction_values_bev] = get_reduced_model_features_n_shap(feature_count_threshold_pv, 
                                                                                            feature_count_threshold_bev, 
                                                                                            run_evaluation, use_normalized=use_normalized)
    
    mean_pv_interactions = np.mean(abs(interaction_values_pv), axis=0)
    mean_bev_interactions = np.mean(abs(interaction_values_bev), axis=0)

    # Plot
    fig, [ax_pv, ax_bev] = plt.subplots(1, 2, figsize=(30, 10))

    plotting.heatmap_interactions(X_red_model_pv, interaction_values_pv, ax=ax_pv, 
                                  feature_name_dict=rename_tick_dict, plot_cbar=True,
                                  fontsize=13, remove_diagonal=True)
    
    plotting.heatmap_interactions(X_red_model_bev, interaction_values_bev, ax=ax_bev,
                                  feature_name_dict=rename_tick_dict, plot_cbar=True,
                                  fontsize=13, remove_diagonal=True)

    # Remove the diagonal

    fig.tight_layout()    

    # Add cartoons
    pv_image = Image.open(f"{__cartoon_path}/pv.png")
    ax_pv_image = fig.add_axes([0.3, .75, 0.3, 0.3])
    ax_pv_image.imshow(pv_image)
    ax_pv_image.axis('off')

    car_image = Image.open(f"{__cartoon_path}/car.png")
    ax_car_image = fig.add_axes([0.8875, .8, 0.16, 0.16])
    ax_car_image.imshow(car_image)
    ax_car_image.axis('off')

    fig.subplots_adjust(right=0.95, wspace=.275)
    if save_fig:
        fig_path = "plots/interaction_heatmaps_pv_and_bev.pdf"
        fig.savefig(fig_path, bbox_inches='tight')

        fig.clear()
        plt.close(fig)

    else:
        plt.show()


    return ax_pv, interaction_values_pv


def plot_decomposed_shap_interactions_pv_and_bev(save_fig: bool = False, 
                                                 run_evaluation: bool = False, 
                                                 feature_count_threshold_pv: int = 15, 
                                                 feature_count_threshold_bev: int = 15,
                                                 target_type: str = 'pv', 
                                                 label_size: float = 14.,
                                                 use_normalized: bool = True):
    """Plot the interaction of different features using interaction values."""

    [X_red_model_pv, shap_values_pv, 
     df_mean_shap_pv, interaction_values_pv,
     X_red_model_bev, shap_values_bev, 
     df_mean_shap_bev, 
     interaction_values_bev] = get_reduced_model_features_n_shap(feature_count_threshold_pv, 
                                                                 feature_count_threshold_bev, 
                                                                 run_evaluation,
                                                                 use_normalized=use_normalized)
    
    if target_type == 'pv':
        X_red_model = X_red_model_pv
        interaction_values = interaction_values_pv
        shap_values = shap_values_pv

        feature = "flats with 5+ rooms"

        feature_inter1 = "share 4-person households"
        feature_inter2 = "global radiation"

    elif target_type == 'bev':
        X_red_model = X_red_model_bev
        interaction_values = interaction_values_bev
        shap_values = shap_values_bev

        feature = "income tax"
        feature_inter1 = "AfD"
        feature_inter2 = "The Greens"

    else:
        raise ValueError("Please choose the target type to be either 'pv' or 'bev'.")

    # Plot
    fig, [ax_tot, ax_main, ax_inter1, ax_inter2] = plt.subplots(1, 4, figsize=(16, 4),
                                                                )

    name_feature = rename_tick_dict[feature] if feature in rename_tick_dict else feature
    plotting.dependence_plot_main_effect(X=X_red_model,
                                         shap_values=shap_values,
                                         feature=feature,
                                         plot_main_effect=False,
                                         y_label="SHAP Values",
                                         x_lim=None, ax=ax_tot,
                                         x_label=name_feature,
                                         font_size=label_size)

    plotting.dependence_plot_main_effect(X=X_red_model,
                                         shap_values=interaction_values,
                                         feature=feature,
                                         plot_main_effect=True,
                                         x_lim=None, ax=ax_main,
                                         x_label=rename_tick_dict[feature] \
                                            if feature in rename_tick_dict else feature,
                                         y_label="SHAP main effect",
                                         font_size=label_size)

    scatter, _ = plotting.dependence_plot_interactions(
            X=X_red_model,
            interaction_vals=interaction_values,
            feature=feature,
            interaction_feature=feature_inter1,
            ax=ax_inter1,
            y_label="SHAP interaction value",
            x_label=rename_tick_dict[feature] \
                        if feature in rename_tick_dict else feature,
            title=rename_tick_dict[feature_inter1] \
                if feature_inter1 in rename_tick_dict else feature_inter1,
            cb_label=None,
            x_lim=None,
            y_lim=None,
            font_size=label_size)

    plotting.dependence_plot_interactions(
            X=X_red_model,
            interaction_vals=interaction_values,
            feature=feature,
            interaction_feature=feature_inter2,
            ax=ax_inter2,
            x_label=rename_tick_dict[feature] \
                        if feature in rename_tick_dict else feature,
            y_label="SHAP interaction value",
            title=rename_tick_dict[feature_inter2] \
                if feature_inter2 in rename_tick_dict else feature_inter2,
            cb_label=None,
            x_lim=None,
            y_lim=None,
            font_size=label_size)

    # Aesthetics
    for ax_r in [ax_tot, ax_main, ax_inter1, ax_inter2]:
        ax_r.axhline(y=0, color='black', linestyle='--', linewidth=1.5)

    fig.tight_layout()

    if "\n" in name_feature:
        addjust_bottom = 0.375
    else:
        addjust_bottom = 0.35
    
    fig.subplots_adjust(bottom=addjust_bottom, 
                        right=0.925, wspace=0.5)

    # Colorbar
    pos_ax_inter1 = ax_inter1.get_position()
    pos_ax_inter2 = ax_inter2.get_position()

    offset_cbar = 0.2
    if "\n" in name_feature:
        offset_cbar += 0.05

    cax = fig.add_axes([pos_ax_inter1.x0, pos_ax_inter2.y0 - offset_cbar, 
                        pos_ax_inter2.x1 - pos_ax_inter1.x0, 
                        0.05])
    cbar = fig.colorbar(scatter, cmap='viridis',
                        cax=cax, orientation='horizontal',
                        ticks=[X_red_model[feature_inter1].min(),
                               X_red_model[feature_inter1].max()])
    #cbar.set_label('values of interacting feature', rotation=0, size=label_size)
    cbar.ax.set_xticklabels(["minimal value of\ninteracting feature", 
                             "maximal value of\ninteracting feature"],)
    cbar.ax.tick_params(labelsize=label_size)
    cbar.solids.set(alpha=1.)

    # Add cartoons

    # Add symbols between the plots
    pos_ax_tot = ax_tot.get_position()
    pos_ax_main = ax_main.get_position()
    middle_y = pos_ax_tot.y0 + (pos_ax_tot.y1 - pos_ax_tot.y0) / 2

    offset_symbol = 0.005
    pos_between_1 = [pos_ax_tot.x1 + offset_symbol, middle_y] 
    pos_between_2 = [pos_ax_main.x1 + offset_symbol, 
                     middle_y]
    pos_between_3 = [pos_ax_inter1.x1 + offset_symbol, 
                     middle_y]
    pos_leftmost = [pos_ax_inter2.x1 + offset_symbol , middle_y]

    fig.text(pos_between_1[0], pos_between_1[1], r"$\boldsymbol{=}$",
             fontsize=30, horizontalalignment='left', verticalalignment='center')

    for pos_r in [pos_between_2, pos_between_3]:
        fig.text(pos_r[0], pos_r[1], r"$\boldsymbol{+}$", 
                 fontsize=30, horizontalalignment='left', verticalalignment='center')

    fig.text(pos_leftmost[0], pos_leftmost[1], r"$\boldsymbol{+}\;\Large{\dots}$", 
             horizontalalignment='left', fontsize=30, verticalalignment='center')

    if save_fig:
        fig_path = f"plots/decomposed_shap_interactions_{target_type}.pdf"
        fig.savefig(fig_path, bbox_inches='tight')

        fig.clear()
        plt.close(fig)

    else:
        plt.show()


    return


def plot_all_figures():

    use_normalized = True

    # Figure 1: Target distribution on a map of Germany.
    #plot_map_distribution_pv_and_bev(save_fig=True)

    # Figure 2
    ## Principle


    # Figure 3: Performance of the recursive GBT
    plot_recursive_gbt_performance(feature_count_threshold_pv=15,
                                   feature_count_threshold_bev=15,
                                   use_normalized=use_normalized,
                                   save_fig=True)

    # Figure 4: Feature importance given by mean absolute Shap values for PV and BEV
    plot_mean_shap_features(feature_count_threshold_pv=15,
                            feature_count_threshold_bev=15,  
                            save_fig=True, run_evaluation=False,
                            use_normalized=use_normalized)
    
    # Figure 5: Feature dependencies for PV and BEV
    plot_dependency_pv_and_bev(save_fig=True, nr_best_features=4,
                               run_evaluation=False, use_normalized=use_normalized)
    
    # Figure 6: Shap interaction for PV
    plot_decomposed_shap_interactions_pv_and_bev(save_fig=True, target_type='pv',
                                                 run_evaluation=False, 
                                                 feature_count_threshold_pv=15, 
                                                 feature_count_threshold_bev=15,
                                                 use_normalized=use_normalized)
    

    ## Appendix
    # Figure A1: All feature dependencies PV
    plot_all_dependencies_separate(feature_count_threshold_pv=15,
                                   feature_count_threshold_bev=15, 
                                   save_fig=True, run_evaluation=False, 
                                   use_normalized=use_normalized)

    # Figure A2: Interaction heatmaps PV and BEV
    plot_interaction_heatmaps_pv_and_bev(save_fig=True, run_evaluation=False,
                                         feature_count_threshold_pv=15, 
                                         feature_count_threshold_bev=15, 
                                         use_normalized=use_normalized)

    # Figure A3: Decomposed SHAP interactions BEV
    plot_decomposed_shap_interactions_pv_and_bev(save_fig=True, target_type='bev',
                                                 run_evaluation=False, 
                                                 feature_count_threshold_pv=15, 
                                                 feature_count_threshold_bev=15,
                                                 use_normalized=use_normalized)

    return