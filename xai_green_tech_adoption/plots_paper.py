#!/usr/bin/env python
# -*- coding: utf-8 -*

"""Plot for producing the figures for the paper 
that presents analysis of the diffusion of both pv and bev."""

import os
import sys

from tqdm import tqdm

import numpy as np
import pandas as pd

import geopandas as gpd

import matplotlib
from matplotlib import pyplot as plt
from PIL import Image

plt.rcParams.update({'text.usetex': True})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

import shap

import pickle
import gzip

__fpath_output = "data/output"
__cartoon_path = "data/raw_data/plotting"


from xai_green_tech_adoption.utils.utils import col_feature_count, ranking_mean_r2_desc, \
    col_run_id, mean_r2_cv_test, mean_r2_cv_train, col_file_path, \
    col_bev_per_vehicle, col_power_accum_pv, col_features, \
    col_idx_train, col_idx_val, col_idx_test, mean_r2_cv_test, col_mean_shap,\
    col_id_ma, col_name_ma, features_norm_to_population_ls, features_norm_drop_ls,\
    col_predictions, col_alpha, col_r2_train, col_r2_test, col_l2_train, col_l2_test,\
    col_mse_train, col_mse_test, col_mae_train, col_mae_test, col_mape_train, col_mape_test,\
    col_count_non_zero, col_std_shap, col_occurences_feat


from xai_green_tech_adoption.shap_analysis import plotting
from xai_green_tech_adoption.shap_analysis.supporting_functions import prepare_performance_dataframe, \
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
                        'CDU/CSU': 'votes CDU/CSU [\\%]',
                        'other parties': 'votes other parties [\\%]',
                        'FDP': 'votes FDP [\\%]',
                        'The Left': 'votes The Left [\\%]',
                        'AfD': 'votes AfD [\\%]',
                        'The Greens': 'votes The Greens [\\%]',
                        'SPD': 'votes SPD [\\%]',
                        'share 4-person households': 'share of\n4-person households [\\%]',
                        'share 5-person households': 'share of\n5-person households [\\%]',
                        'flats with 5+ rooms': 'share of\nflats with 5+ rooms [\\%]',
                        col_power_accum_pv: 'photovoltaic power [kWp/hh]',
                        'global radiation': 'global radiation [kWh/m²]',
                        'income tax': 'income tax [€ p.c.]',}

def load_performance_and_metadata_dataframes(use_normalized: bool = True):

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
                                 use_normalized: bool = False) -> list:
    """Get detailed information about the model given by by 'split_id_best_model'.

    Args:
        df_perf (pd.DataFrame): _description_
        df_metadata (pd.DataFrame): _description_
        model_dict (dict): _description_
        split_id_best_model (int): _description_
        feature_count_threshold (int): _description_
        use_normalized (bool, optional): _description_. Defaults to False.

    Returns:
        list: Features of the reduced model, the reduced model,
             SHAP values and SHAP interaction values.
    """

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
                                     label_size=40):
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

    cbar.ax.set_xticklabels(["minimal value of\ntarget", 
                             "maximal value of\ntarget"],)
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


def plot_separate_parts_method_figure(save_fig: bool = False,
                      cmap='inferno', use_normalized: bool = True, 
                      feature_count_threshold_pv: int = 15):
    """Plot the parts of the figure explaining the approach"""

    # Load target data
    data_dtype_dict = {col_id_ma: int}
    df_data_pv = pd.read_csv("data/input/input.csv", sep=';', 
                             dtype=data_dtype_dict)
    #df_data_pv[col_id_ma] = df_data_pv[col_id_ma].str.pad(9, fillchar="0")
    
    # Target + Input Map
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
    [df_perf_pv, _, 
     df_meta_pv, _] = load_performance_and_metadata_dataframes(use_normalized=use_normalized)

    
    
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
    
    [X_red_model_pv, shap_values_pv, _, interaction_values_pv,
     _, _, _, _] = get_reduced_model_features_n_shap(feature_count_threshold_pv=15, 
                                                     feature_count_threshold_bev=15, 
                                                     run_evaluation=False,
                                                     use_normalized=use_normalized)
    
    red_model = model_dict_pv[split_id_best_model]
    y_pred = red_model.predict(X_red_model_pv)
    df_pred = pd.DataFrame({col_id_ma: df_data_pv[col_id_ma], col_predictions: y_pred})
    tree_explainer = shap.explainers.Tree(red_model)    
    shap_values = tree_explainer.shap_values(X_red_model_pv)

    # Map with predictions
    fig_map_predict = plt.figure(figsize=(10, 10))
    ax_pred = fig_map_predict.add_subplot(111)
    gdf_w_pred = pd.merge(left=gdf_complete_pv,right=df_pred, on=col_id_ma, how='left')
    gdf_w_pred.plot(column=col_predictions, ax=ax_pred, cmap=cmap)

    # Map with differences
    gdf_w_pred['diff'] = abs(gdf_w_pred[col_power_accum_pv] - gdf_w_pred[col_predictions])
    fig_map_diff, ax_map_diff = plt.subplots(figsize=(10, 10))
    gdf_w_pred.plot(column='diff', ax=ax_map_diff, cmap='viridis')


    # Plot for a specific city (picked a random example )
    print("Showing prediction for " + 
          f"{df_data_pv.loc[df_data_pv[col_id_ma] == 53340020][col_name_ma].values}")
    idx_city = df_data_pv.loc[df_data_pv[col_id_ma] == 53340020].index
    shap_city = shap_values[idx_city, :]
    shap_city = shap_city[0]
    idx_sorted = np.argsort(np.abs(shap_city))[::-1]

    mean_plus_shap = df_pred[col_predictions].mean()

    fig_shap_addition = plt.figure(figsize=(7, 4))
    ax_shap_addition = fig_shap_addition.add_subplot(111)

    nn_features_shown = 3
    for idx, ranking_feature in enumerate(range(nn_features_shown)):
        if shap_city[idx_sorted[ranking_feature]] >= 0:
            box = ax_shap_addition.broken_barh(
                [(mean_plus_shap, shap_city[idx_sorted[ranking_feature]])],
                (11 + idx * 10, 7),
                facecolors=("firebrick"),
                zorder=2,
            )
            mean_plus_shap += shap_city[idx_sorted[ranking_feature]]

        else:
            box = ax_shap_addition.broken_barh(
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
        print(f"\tChange SHAP: {shap_city[idx_sorted[ranking_feature]]:.3f}")

        # Add values to bars
        value_str = f"{shap_city[idx_sorted[ranking_feature]]:.2f}"
        if shap_city[idx_sorted[ranking_feature]] > 0:
            value_str = "+" + value_str

        ax_shap_addition.text(mean_plus_shap - shap_city[idx_sorted[ranking_feature]] / 2, 
                              11 + idx * 10 + 3.75,
                              value_str, 
                              fontsize=30, 
                              color='white', 
                              horizontalalignment='center',
                              verticalalignment='center')
        
        ax_shap_addition.vlines(x=mean_plus_shap, ymin=11 + idx * 10,
                                ymax=11 + (idx + 1) * 10 + 7, color='k', zorder=4,
                                linewidth=3.4)

    if nn_features_shown < feature_count_threshold_pv:
        shap_remaining = sum(shap_city[idx_sorted[nn_features_shown:]])
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
            X_red_model_pv.columns[idx_sorted[:nn_features_shown]].to_list() + ["remaining features"]
        )
        
        mean_plus_shap += shap_remaining
    else:
        ax_shap_addition.set_ylim(5, (nn_features_shown + 1) * 10)
        ax_shap_addition.set_yticks(
            range(15, 15 + nn_features_shown * 10, 10),
            labels=[
                X_red_model_pv.columns[ranking_feature] 
                for ranking_feature in idx_sorted[:nn_features_shown]
            ],
        )
    
    ax_shap_addition.invert_yaxis()
    
    plt.axvline(x=np.mean(y_pred), color="darkgray", zorder=1, linewidth=3)
    plt.axvline(x=mean_plus_shap, color="darkgray", ls="--", zorder=1, linewidth=3)
    
    ax_shap_addition.spines[["right", "bottom", "top", "left"]].set_visible(False)

    ax_shap_addition.text(np.mean(y_pred)+.004, 1.05, "$\\boldsymbol{\\phi_0}$", fontsize=40,
                            horizontalalignment='center')
    ax_shap_addition.text(mean_plus_shap, 1.05, "$\\boldsymbol{f(x)}$", fontsize=40,
                           horizontalalignment='center')
    ax_shap_addition.set_xlabel("predicted power of\nPV systems installed", fontsize=20)

    ax_shap_addition.tick_params(axis="both", which="both", labelsize=20)
    ax_shap_addition.tick_params(left=False)
    ax_shap_addition.tick_params(axis="x", which="both", 
                                 bottom=False, top=False, labelbottom=False)

    # Feature importance
    fig_shap_importance, ax_shap_imp = plt.subplots(figsize=(7, 4))
    feature_names = X_red_model_pv.columns[idx_sorted[:nn_features_shown]].to_list()
    plotting.bar_shap_reduced(X_red_model_pv, shap_values, bar_color='royalblue',
                              show_std=False, ax=ax_shap_imp, n_print=nn_features_shown)
    ax_shap_imp.set_yticks(range(nn_features_shown), 
                          labels=feature_names)
    ax_shap_imp.set_xlabel("mean $|\\text{SHAP}|$", fontsize=20)

    # Dependencies
    fig_dep, ax_dep = plt.subplots(figsize=(6, 4))
    feat_r = X_red_model_pv.iloc[:, idx_sorted[0]].name
    xlabel_feat = rename_tick_dict[feat_r] if feat_r in rename_tick_dict else feat_r
    plotting.dependence_plot(X_red_model_pv, shap_values_pv, feature=feat_r,
                             ax=ax_dep, x_label=xlabel_feat, y_label="SHAP values",)
    ax_dep.axhline(y=0, color='k', ls='--', lw=2)

    # Aesthetics
    map_axis = [ax_target, ax_input, ax_pred, ax_map_diff]
    for ax_r in map_axis:
        ax_r.axis('off')
        ax_r.set_rasterized(True)

    fig_shap_addition.tight_layout()
    fig_shap_importance.tight_layout()

        

    if save_fig:
        fpath_fig_target = "plots/method_target_map.svg"
        fpath_fig_input = "plots/method_input_map.svg"
        fpath_fig_pred = "plots/method_prediction_map.svg"
        fpath_fig_diff = "plots/method_absdiff_target_pred_map.svg"

        fpath_shap_addition = "plots/method_shap_addition.svg"
        fpath_feature_importance = "plots/method_feature_importance.svg"
        fpath_dep = "plots/method_dependency.svg"

        fig_target.savefig(fpath_fig_target, bbox_inches='tight', transparent=True)
        fig_input.savefig(fpath_fig_input, bbox_inches='tight', transparent=True)
        fig_map_predict.savefig(fpath_fig_pred, bbox_inches='tight', transparent=True)
        fig_map_diff.savefig(fpath_fig_diff, bbox_inches='tight', transparent=True)

        fig_shap_addition.savefig(fpath_shap_addition, bbox_inches='tight', transparent=True)
        fig_shap_importance.savefig(fpath_feature_importance, bbox_inches='tight', 
                                    transparent=True)
        fig_dep.savefig(fpath_dep, bbox_inches='tight', 
                        transparent=True)

        for fig_r in [fig_target, fig_input, 
                      fig_map_predict, fig_map_diff,
                      fig_shap_addition, fig_shap_importance,
                      fig_dep]:
            fig_r.clear()
            plt.close(fig_r)

    else:

        plt.show()

    return


def plot_recursive_gbt_performance(feature_count_threshold_pv: int = 15, 
                                   feature_count_threshold_bev: int = 15,
                                   save_fig: bool = False, 
                                   use_normalized: bool = True,
                                   poster_version: bool = False):

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
        fig_path = "plots/rfe_performance_pv_and_bev"
        if poster_version:
            fig.savefig(fig_path + "_poster.png", bbox_inches='tight',
                        transparent=True)
        else:
            fig.savefig(fig_path + ".pdf", bbox_inches='tight')

        fig.clear()
        plt.close(fig)

    else:
        plt.show()

    return


def plot_mean_shap_features(feature_count_threshold_pv: int = 15,
                            feature_count_threshold_bev: int = 15,  
                            save_fig: bool = False, nr_features_shown: int = 25,
                            fig_height: int = 17,
                            run_evaluation: bool = False, labelsize: int = 18,
                            use_normalized: bool = True, 
                            poster_version: bool = False):
    
    if poster_version:
        print("Figure for poster with hard coded height, fontsize and number of features shown.")
        fig_height = 10
        nr_features_shown = 15
        labelsize = 25

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
    fig, [ax_pv, ax_bev] = plt.subplots(1, 2, figsize=(20, fig_height))
    plotting.bar_shap_feature_imp(df_mean_shap_pv, features_pv,
                                  ax=ax_pv, nr_features_shown=nr_features_shown,
                                  feature_occurence_lims=feature_occ_lims,
                                   cmap='viridis', labelsize=labelsize,
                                   xlabel="mean $|\\text{SHAP values}|$ [kWp/hh]")
    sm = plotting.bar_shap_feature_imp(df_mean_shap_bev, features_bev, 
                                  ax=ax_bev, nr_features_shown=nr_features_shown,
                                  feature_occurence_lims=feature_occ_lims,
                                  cmap='viridis', labelsize=labelsize,
                                  xlabel="mean $|\\text{SHAP values}|$ [\\%]",
                                  scalefactor_x=100.)
    
    fig.tight_layout()

    cax = fig.add_axes([.925, .5, 0.02 , .3])
    cbar_ticks = np.array([1, 4, 7, 10]) 
    cbar = fig.colorbar(sm, cax=cax, ticks=cbar_ticks + .5)
    cbar.set_label('number feature occurs in runs', rotation=270, labelpad=25, size=labelsize,
                   )
    cbar.ax.tick_params(labelsize=labelsize *1.3)
    cbar.ax.tick_params(which='minor', length=0)
    cbar.set_ticklabels(cbar_ticks)

    # Add cartoons
    pv_image = Image.open(f"{__cartoon_path}/pv.png")
    ax_pv_image = fig.add_axes([0.25, .1, 0.25, 0.25])
    ax_pv_image.imshow(pv_image)
    ax_pv_image.axis('off')

    car_image = Image.open(f"{__cartoon_path}/car.png")
    if poster_version:
        ax_car_image = fig.add_axes([0.775, .12, 0.2, 0.2])
    else:
        ax_car_image = fig.add_axes([0.775, .1, 0.2, 0.2])

    ax_car_image.imshow(car_image)
    ax_car_image.axis('off')

    if save_fig:
        fig_path = "plots/shap_feature_importance_pv_n_bev"

        if poster_version:
            fig.savefig(fig_path + "_poster.png", bbox_inches='tight',
                        transparent=True)
        else:
            fig.savefig(fig_path + ".pdf", bbox_inches='tight')

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
        shap_values_pv, 
        interaction_values_pv] = get_details_about_best_model(df_perf_pv, df_metadata_pv, 
                                                              model_dict_pv, 
                                                              split_id_best_model, 
                                                              feature_count_threshold_pv)

        
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


def plot_dependency_pv_and_bev(feature_count_threshold_pv: int = 15,
                              feature_count_threshold_bev: int=15, 
                              save_fig: bool = False, nr_best_features: int = 4, 
                              sort_method: str = "model", run_evaluation: bool = False, use_normalized: bool = True,
                              poster_version: bool = False):
    """Plot 2 by 4 with depency plot in the best model.

    Args:
        feature_count_threshold_pv (int, optional): Number of features in reduced model for pv. Defaults to 15.
        feature_count_threshold_bev (int, optional): Number of features in reduced model for pv. Defaults to 15.
        save_fig (bool, optional): If 'True', save figure. Defaults to False.
        nr_best_features (int, optional): Number of best features shown. Defaults to 4.
        run_evaluation (bool, optional): if 'True', run the evaluation again. Defaults to False.
        use_normalized (bool, optional): If 'True', run the version with additionally normalized to
        population features. Defaults to True.
        poster_version (bool, optional): if 'True', change figure sizes to produce a figure
        for a poster. Defaults to False.
    """

    if sort_method not in ["mean", "model"]:
        raise IOError("sort_method must be either 'mean' or 'model'.")

    [X_red_model_pv, shap_values_pv, df_mean_shap_pv, _,
     X_red_model_bev, shap_values_bev, df_mean_shap_bev, _] = get_reduced_model_features_n_shap(feature_count_threshold_pv, 
                                                                                            feature_count_threshold_bev, 
                                                                                            run_evaluation,
                                                                                            use_normalized=use_normalized)

    #return X_red_model_pv, shap_values_pv

    # Plot
    fig, [ax_pv, ax_bev] = plt.subplots(2, nr_best_features, 
                                        figsize=(5*nr_best_features, 10),                    
                                        sharey='row')

    ## PV row
    ## Choose list of best features
    if sort_method == "mean":
        ranking_features_pv = df_mean_shap_pv.loc[:, 
                                                  col_mean_shap].drop(mean_r2_cv_test, 
                                                                  axis=0)
        assert np.diff(ranking_features_pv.values).min() >= 0, "Not sorted."
        sorted_feature_list_pv = list(ranking_features_pv.index)[::-1]
        
    elif sort_method == "model":
        sort_idx_shap = np.argsort(np.mean(abs(shap_values_pv), axis=0))
        sorted_feature_list_pv = list(X_red_model_pv.columns[sort_idx_shap])[::-1]
        
    count_plotted = 0

    for feat_r in sorted_feature_list_pv:
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
    if sort_method == "mean":
        ranking_features_bev = df_mean_shap_bev.loc[:, 
                                                  col_mean_shap].drop(mean_r2_cv_test, 
                                                                  axis=0)
        assert np.diff(ranking_features_bev.values).min() >= 0, "Not sorted."
        sorted_feature_list_bev = list(ranking_features_bev.index)[::-1]
        
    elif sort_method == "model":
        sort_idx_shap = np.argsort(np.mean(abs(shap_values_bev), axis=0))
        sorted_feature_list_bev = list(X_red_model_bev.columns[sort_idx_shap])[::-1]

    count_plotted = 0

    for feat_r in  sorted_feature_list_bev:
        if feat_r in X_red_model_bev.columns:
            xlabel_feat = rename_tick_dict[feat_r] if feat_r in rename_tick_dict else feat_r
            plotting.dependence_plot(X_red_model_bev, shap_values_bev,
                                     feature=feat_r,
                                     x_label=xlabel_feat,
                                     y_label="", label_size=20,
                                     ax=ax_bev[count_plotted],
                                     scalefactor_y=100)
            
            count_plotted += 1

        else:
            print(f"Feature {feat_r} not in reduced model.")

        if count_plotted == nr_best_features:
            break

    # Aesthetics
    ## Adjust xticks of BEV    
    ax_pv[0].set_ylabel("SHAP values [kWp/hh]", size=20)  
    ax_bev[0].set_ylabel("SHAP values [\\%]", size=20)

    for ax_r in np.concatenate((ax_pv, ax_bev)):
        ax_r.axhline(y=0, color='black', linestyle='--', linewidth=1.5)

    fig.tight_layout()
    fig.subplots_adjust(right=0.925, hspace=.4, top=.9)
    # Add cartoons
    pv_image = Image.open(f"{__cartoon_path}/pv.png")
    ax_pv_image = fig.add_axes([0.815, .775, 0.22, 0.22])
    ax_pv_image.imshow(pv_image)
    ax_pv_image.axis('off')


    car_image = Image.open(f"{__cartoon_path}/car.png")
    ax_car_image = fig.add_axes([0.875, .33, 0.122, 0.122])
    ax_car_image.imshow(car_image)
    ax_car_image.axis('off')


    if save_fig:
        
        if sort_method == "mean":
            fpath_postfix = "_mean"

        elif sort_method == "model":
            fpath_postfix = ""

        if poster_version:
            fig_path = f"plots/dependency_plots_pv_and_bev_poster{fpath_postfix}.png"
            fig.savefig(fig_path, bbox_inches='tight', transparent=True)
        else:
            fig_path = f"plots/dependency_plots_pv_and_bev{fpath_postfix}.pdf"
            fig.savefig(fig_path, bbox_inches='tight')

        fig.clear()
        plt.close(fig)

    else:
        plt.show()

    return


def plot_all_dependencies_separate(feature_count_threshold_pv: int = 15,
                                   feature_count_threshold_bev: int = 15,
                                   save_fig: bool = False, run_evaluation: bool = False,
                                   labelsize: float=16., use_normalized: bool = True):
    """Plot shap values over the feature values for all features of the best number.

    Args:
        feature_count_threshold_pv (int, optional): Number of features used for the reduced 
            pv model. Defaults to 15.
        feature_count_threshold_bev (int, optional): Number of features used for the reduced 
            bev model. Defaults to 15.
        save_fig (bool, optional): If 'True', save figure. Defaults to False.
        run_evaluation (bool, optional): If 'True' the evaluation is run again. Defaults to False.
        labelsize (float, optional): Fontsize of figure labels. Defaults to 16..
        use_normalized (bool, optional): If 'True', use version where remaining features 
            where normed to population. Defaults to True.
    """


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
                                ax=ax_arr_bev.flatten()[idx_plot],
                                scalefactor_y=100.)
        ax_arr_bev.flatten()[idx_plot].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        idx_plot += 1
        
    if idx_plot < len(ax_arr_bev.flatten()):
        for idx_r in range(idx_plot, len(ax_arr_bev.flatten())):
            ax_arr_bev.flatten()[idx_r].axis('off')

    # Aesthetics
    for ax_r in ax_arr_pv[:,0].flatten():
        ax_r.set_ylabel("SHAP values [kWp/hh]", size=labelsize)

    for ax_r in ax_arr_bev[:,0].flatten():
        ax_r.set_ylabel("SHAP values [\\%]", size=labelsize)

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
    """Plot heatmap of interatction shap interaction values for both pv and bev for
    the best performing model with 'feature_count_threshold-...' features.

    Args:
        feature_count_threshold_pv (int, optional): Number of features of the 
            reduced pv model. Defaults to 15.
        feature_count_threshold_bev (int, optional): Number of features of the 
            reduced pv model. Defaults to 15.
        save_fig (bool, optional): If 'True', save figure. Defaults to False.
        run_evaluation (bool, optional): If 'True' . Defaults to False.
        use_normalized (bool, optional): _description_. Defaults to True.
    """

    # Load data

    [X_red_model_pv, shap_values_pv, df_mean_shap_pv, interaction_values_pv,
     X_red_model_bev, shap_values_bev, df_mean_shap_bev, interaction_values_bev] = get_reduced_model_features_n_shap(feature_count_threshold_pv, 
                                                                                            feature_count_threshold_bev, 
                                                                                            run_evaluation, use_normalized=use_normalized)
    
    #return X_red_model_pv, interaction_values_pv
    mean_pv_interactions = np.mean(abs(interaction_values_pv), axis=0)
    mean_bev_interactions = np.mean(abs(interaction_values_bev), axis=0)

    # Plot
    fig, [ax_pv, ax_bev] = plt.subplots(1, 2, figsize=(30, 10))

    plotting.heatmap_interactions(X_red_model_pv, interaction_values_pv, ax=ax_pv, 
                                  feature_name_dict=rename_tick_dict, plot_cbar=True,
                                  fontsize=13, remove_diagonal=True,
                                  cbar_label='mean $|\\text{SHAP interaction value}|$ [kWp/hh]')
    
    plotting.heatmap_interactions(X_red_model_bev, interaction_values_bev, ax=ax_bev,
                                  feature_name_dict=rename_tick_dict, plot_cbar=True,
                                  fontsize=13, remove_diagonal=True, scalefactor_cbar=100.,
                                  cbar_label='mean $|\\text{SHAP interaction value}|$ [\\%]')

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


    return


def plot_decomposed_shap_interactions_pv_and_bev(save_fig: bool = False, 
                                                 run_evaluation: bool = False, 
                                                 feature_count_threshold_pv: int = 15, 
                                                 feature_count_threshold_bev: int = 15,
                                                 target_type: str = 'pv', 
                                                 label_size: float = 14.,
                                                 use_normalized: bool = True,
                                                 poster_version: bool = False):
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

        unit_str = " [kWp/hh]"

        scale_factor = 1.

    elif target_type == 'bev':
        X_red_model = X_red_model_bev
        interaction_values = interaction_values_bev
        shap_values = shap_values_bev

        feature = "income tax"
        feature_inter1 = "AfD"
        feature_inter2 = "The Greens"

        unit_str = " [\\%]"

        scale_factor = 100.

    else:
        raise ValueError("Please choose the target type to be either 'pv' or 'bev'.")

    # Plot
    if poster_version:
        fig, [[ax_tot, ax_main], 
              [ax_inter1, ax_inter2]] = plt.subplots(2, 2, figsize=(10, 8),
                                                                )

    else:
        fig, [ax_tot, ax_main, ax_inter1, ax_inter2] = plt.subplots(1, 4, 
                                                                    figsize=(16, 4),
                                                                )

    name_feature = rename_tick_dict[feature] if feature in rename_tick_dict else feature
    if poster_version:
        name_feature = name_feature.replace("\n", " ")
    plotting.dependence_plot_main_effect(X=X_red_model,
                                         shap_values=shap_values * scale_factor,
                                         feature=feature,
                                         plot_main_effect=False,
                                         y_label="SHAP values" + unit_str,
                                         x_lim=None, ax=ax_tot,
                                         x_label=name_feature,
                                         font_size=label_size)

    plotting.dependence_plot_main_effect(X=X_red_model,
                                         shap_values=interaction_values * scale_factor,
                                         feature=feature,
                                         plot_main_effect=True,
                                         x_lim=None, ax=ax_main,
                                         x_label=name_feature,
                                         y_label="SHAP main effect" + unit_str,
                                         font_size=label_size)

    name_feat_inter1 = rename_tick_dict[feature_inter1] \
                if feature_inter1 in rename_tick_dict else feature_inter1
    if poster_version:
        name_feat_inter1 = name_feat_inter1.replace("\n", " ")
    scatter, _ = plotting.dependence_plot_interactions(
            X=X_red_model,
            interaction_vals=interaction_values * scale_factor,
            feature=feature,
            interaction_feature=feature_inter1,
            ax=ax_inter1,
            y_label="SHAP interaction value" + unit_str,
            x_label=name_feature,
            title=name_feat_inter1,
            cb_label=None,
            x_lim=None,
            y_lim=None,
            font_size=label_size)

    name_feat_inter2 = rename_tick_dict[feature_inter2] \
                if feature_inter2 in rename_tick_dict else feature_inter2
    plotting.dependence_plot_interactions(
            X=X_red_model,
            interaction_vals=interaction_values * scale_factor,
            feature=feature,
            interaction_feature=feature_inter2,
            ax=ax_inter2,
            x_label=name_feature,
            y_label="SHAP interaction value" + unit_str,
            title=name_feat_inter2,
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

    if poster_version:
        fig.subplots_adjust(bottom=.2, 
                        right=0.875, wspace=0.5, hspace=0.5)
    else:
        fig.subplots_adjust(bottom=addjust_bottom, 
                            right=0.925, wspace=0.6)
    # Colorbar
    pos_ax_inter1 = ax_inter1.get_position()
    pos_ax_inter2 = ax_inter2.get_position()

    offset_cbar = 0.2
    if "\n" in name_feature:
        offset_cbar += 0.05

    if poster_version:
        offset_cbar = 0.125

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
    middle_y_2 = pos_ax_inter1.y0 + (pos_ax_inter1.y1 - pos_ax_inter1.y0) / 2
    
    if poster_version:
        offset_symbol = 0.01
    else:
        offset_symbol = 0.005
    pos_between_1 = [pos_ax_tot.x1 + offset_symbol, middle_y] 
    pos_between_2 = [pos_ax_main.x1 + offset_symbol, 
                     middle_y]
    pos_between_3 = [pos_ax_inter1.x1 + offset_symbol, 
                     middle_y_2]
    pos_leftmost = [pos_ax_inter2.x1 + offset_symbol , middle_y_2]

    fig.text(pos_between_1[0], pos_between_1[1], r"$\boldsymbol{=}$",
             fontsize=30, horizontalalignment='left', verticalalignment='center')

    for pos_r in [pos_between_2, pos_between_3]:
        fig.text(pos_r[0], pos_r[1], r"$\boldsymbol{+}$", 
                 fontsize=30, horizontalalignment='left', verticalalignment='center')

    fig.text(pos_leftmost[0], pos_leftmost[1], r"$\boldsymbol{+}\;\Large{\dots}$", 
             horizontalalignment='left', fontsize=30, verticalalignment='center')

    if save_fig:
        fig_path = f"plots/decomposed_shap_interactions_{target_type}"

        if poster_version:
            fig_path += "_poster"
        fig.savefig(fig_path + ".pdf", bbox_inches='tight', 
                    transparent=True)

        fig.clear()
        plt.close(fig)

    else:
        plt.show()


    return


def lasso_get_feature_list(df_lasso: pd.DataFrame):


    feature_list = list(df_lasso.columns)
    for col_no_feat in [
        col_run_id,
        col_alpha,
        col_r2_train,
        col_r2_test,
        col_l2_train,
        col_l2_test,
        col_mse_train,
        col_mse_test,
        col_mae_train,
        col_mae_test,
        col_mape_train,
        col_mape_test,
    ]:
        feature_list.remove(col_no_feat)


    return feature_list


def plot_performance_large_alpha(save_fig: bool = False, alpha_inv_max = 10100):
    """Plot the performance for large alpha values indicated by mean R2 score, mean mae
    and number of non-zero coefficients."""

    # Load data
    fpath_data = "data/output/benchmarking_lasso_large_alpha_norm.pklz"
    df_lasso = pd.read_pickle(fpath_data, compression='gzip')

    feature_list = lasso_get_feature_list(df_lasso)
    df_lasso[col_count_non_zero] = (df_lasso[feature_list] != 0).sum(axis=1)
    df_lasso_mean_perf = pd.DataFrame(df_lasso.groupby(by=col_alpha, 
                                                       as_index=False).mean())
    
    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex='all')

    plotting.plot_performance_lasso(df_lasso_mean_perf, 
                                    perf_metric_train=col_r2_train,
                                    perf_metric_test=col_r2_test, ax=ax[0],
                                    x_max=alpha_inv_max)
    
    plotting.plot_performance_lasso(df_lasso_mean_perf, perf_metric_train=col_mae_train,
                                    perf_metric_test=col_mae_test, ax=ax[1],
                                    x_max=alpha_inv_max)
    
    plotting.plot_count_non_zero_coef(df_lasso_mean_perf, col_alpha,
                                      x_max=alpha_inv_max, ax=ax[2],
                                      color="#d95f02")
    

    # Aesthetics
    ax[0].set_ylim(0.5, 0.9)
    ax[1].set_ylim(0.1, 0.14)
    ax[2].set_ylim(0, 170)
    ax[0].set_xlim(0, alpha_inv_max)

    legend = ax[0].legend(numpoints=None, fontsize=20)
    for leg_hdls in legend.legend_handles:
        leg_hdls._sizes = [80]

    ax[0].set_ylabel("mean $R^2$ score", size=20)
    ax[1].set_ylabel("mean MAE", size=20)
    ax[2].set_ylabel("number of features\nwith non-zero coefficients", size=20)
    ax[-1].set_xlabel("$\\alpha^{-1}$", size=26)

    for ax_r in ax:
        ax_r.tick_params(labelsize=14)
    
    label_ls = ["(a)", "(b)", "(c)", "(d)"]
    for idx, ax_r in enumerate(ax):
        ax_r.text(-0.15, 1.1, label_ls[idx], 
                  transform=ax[idx].transAxes, fontsize=20)


    if save_fig:
        fig_path = "plots/SI_lasso_performance_large_alpha.pdf"
        fig.savefig(fig_path, bbox_inches='tight')

        fig.clear()
        plt.close(fig)

    else:
        plt.show()

    return


def plot_performance_and_coefficients_small_alpha(save_fig: bool = False, 
                                                  alpha_inv_max: float = 44,
                                                  threshold_inverse_alpha: int = 38):
    """Plot the coefficients of the benchmark model that uses linear
    regression."""

    # Load data
    fpath_data = "data/output/benchmarking_lasso_norm.pklz"
    df_lasso = pd.read_pickle(fpath_data, compression='gzip')

    feature_list = lasso_get_feature_list(df_lasso)
    df_lasso[col_count_non_zero] = (df_lasso[feature_list] != 0).sum(axis=1)
    df_lasso_mean_perf = pd.DataFrame(df_lasso.groupby(by=col_alpha, 
                                                       as_index=False).mean())
    
    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex='all')
    fig_coeff, ax_coeff = plt.subplots(figsize=(10, 6))

    plotting.plot_performance_lasso(df_lasso_mean_perf, 
                                    perf_metric_train=col_r2_train,
                                    perf_metric_test=col_r2_test, ax=ax[0],
                                    x_max=alpha_inv_max)

    plotting.plot_count_non_zero_coef(df_lasso_mean_perf, col_alpha,
                                      x_max=alpha_inv_max, ax=ax[1],
                                      color="#d95f02")
    

    label_colors, color_dict = plotting.plot_mean_coefficients(df_lasso_mean_perf, 
                                                   feature_list=feature_list,
                                    ax=ax_coeff, x_max=alpha_inv_max, 
                                    threshold_inv_alpha=threshold_inverse_alpha, 
                                    feature_rename_dict=rename_tick_dict)
    
    for ax_r in ax:
        ax_r.axvline(x=threshold_inverse_alpha, 
                     color='darkred', linestyle='--', linewidth=2.)
    ax_coeff.axvline(x=threshold_inverse_alpha,
                     color='darkred', linestyle='--', linewidth=2.)
                     
    # Aesthetics
    ## Performance plot
    legend = ax[0].legend(numpoints=None, fontsize=20)
    for leg_hdls in legend.legend_handles:
        leg_hdls._sizes = [80]

    ax[0].set_ylabel("mean $R^2$ score", size=20)
    ax[1].set_ylabel("number of features\nwith non-zero coefficients", size=20)
    ax[-1].set_xlabel("$\\alpha^{-1}$", size=26)

    for ax_r in ax:
        ax_r.tick_params(labelsize=14)

    ## Coeff. Plot
    ax_coeff.set_ylabel("mean $\\beta_i$", size=26)
    ax_coeff.set_xlabel("$\\alpha^{-1}$", size=26)
    ax_coeff.tick_params(labelsize=14)
    
    # Legend outside plots
    legend_coeff = ax_coeff.legend(numpoints=None, 
                                   fontsize=10, bbox_to_anchor=(1.01, 1.05), 
                   loc='upper left', markerscale=10, labelcolor=label_colors)
    
    fig_coeff.tight_layout()
    fig_coeff.subplots_adjust(right=0.725)
    
    
    label_ls = ["(a)", "(b)", "(c)", "(d)"]
    for idx, ax_r in enumerate(ax):
        ax_r.text(-0.15, 1.1, label_ls[idx], 
                  transform=ax[idx].transAxes, fontsize=20)
        
    ax_coeff.text(-.075, 1.0175, label_ls[2], 
                  transform=ax_coeff.transAxes, fontsize=20)
        
    if save_fig:
        fpath_fig = "plots/SI_lasso_performance_small_alpha.pdf"

        fig.savefig(fpath_fig, bbox_inches='tight')
        fig.clear()
        plt.close(fig)

        fpath_fig_coeff = f"plots/SI_lasso_coefficients_small_alpha_invalphathres{threshold_inverse_alpha}.pdf"
        fig_coeff.savefig(fpath_fig_coeff, bbox_inches='tight')
        fig_coeff.clear()
        plt.close(fig_coeff)

    else:
        plt.show()


    return
    

def plot_benchmark_shap_feature_importance(save_fig: bool = False, 
                                           alpha_lasso: float = 1./38.,#0.026324,
                                           normed: bool = True, 
                                           labelsize: float = 18,
                                           show_progress: bool = False):

    fpath_lasso = "data/output/benchmarking_lasso_norm.pklz"
    df_lasso = pd.read_pickle(fpath_lasso, 
                              compression='gzip')
    
    unique_alphas = df_lasso[col_alpha].unique()
    alpha_closest_idx =  np.argmin(abs(unique_alphas - alpha_lasso))
    alpha_chosen = unique_alphas[alpha_closest_idx]

    print(f"Wanted alpha={alpha_lasso}, " + 
          f"chosen alpha={alpha_chosen} (alpha_inv:{1./alpha_chosen:.8f}, diff: {alpha_chosen - alpha_lasso})")
    
    feature_list = lasso_get_feature_list(df_lasso)

    # Load data
    df_metadata = pd.read_csv("data/output/metadata_rfe_norm.csv", sep=";")
    df_metadata = prepare_metadata_dataframe(df_metadata, ["indices_training_set", 
                                                           "indices_val_set", 
                                                           "indices_test_set"])
    
    df_data = pd.read_csv(df_metadata[col_file_path].unique()[0], sep=";")
    df_data.loc[
    df_data["mean distance public transport"].isin([np.inf, -np.inf]),
    "mean distance public transport",
                ] = df_data.loc[
    ~df_data["mean distance public transport"].isin([np.inf, -np.inf]),
    "mean distance public transport",
    ].max()
    if normed:
        norm_ls = features_norm_to_population_ls
        drop_ls = features_norm_drop_ls

        df_data = df_data.drop(columns=drop_ls)

        for feat_r in norm_ls:
            feat_new = feat_r + "_per_capita"
            df_data[feat_new] = df_data[feat_r] / df_data['population'].astype(float)
            df_data.drop(columns=feat_r, inplace=True)

    # Compute mean shap feature importance for all training-test-splits
    df_mean_shap = pd.DataFrame()
    df_std_shap = pd.DataFrame()

    for split in tqdm(df_metadata[col_run_id], disable=not show_progress):
        idx_train = df_metadata.loc[df_metadata[col_run_id] == split, col_idx_train].values[0]
        idx_val = df_metadata.loc[df_metadata[col_run_id] == split, col_idx_val].values[0]



        df_mean_split = pd.DataFrame(df_data[feature_list].iloc[idx_train + idx_val].mean()).T
        df_std_split = pd.DataFrame(df_data[feature_list].iloc[idx_train + idx_val].std(ddof=0)).T

        df_coef = df_lasso.loc[(df_lasso[col_alpha] == alpha_chosen) &
                                (df_lasso[col_run_id] == split), feature_list,]
        df_shap = df_coef.iloc[0] * ((df_data[feature_list] - df_mean_split.iloc[0]) / 
                                     df_std_split.iloc[0])

        # compute shap feature importances and std of abs of shap values
        df_mean_shap_split = pd.DataFrame(df_shap.abs().mean(axis=0)).T
        df_std_shap_split = pd.DataFrame(df_shap.abs().std(ddof=0, axis=0)).T
        df_mean_shap = pd.concat([df_mean_shap, df_mean_shap_split])
        df_std_shap = pd.concat([df_std_shap, df_std_shap_split])
    
    df_fi = pd.DataFrame(
        {
        col_mean_shap: df_mean_shap[feature_list].mean(axis=0),
        col_std_shap: df_std_shap[feature_list].std(ddof=0, axis=0),
        col_occurences_feat: df_mean_shap[feature_list].astype(bool).sum(axis=0),
        }
    )

    df_fi = df_fi.loc[df_fi[col_mean_shap] != 0]
    df_fi = df_fi.sort_values(col_mean_shap)
    
    # Plot

    fig, ax = plt.subplots(figsize=(8, 10))

    sm = plotting.bar_shap_feature_imp(df_fi, df_fi.index, ax=ax, cmap='viridis', 
                                  xlabel="mean $|\\text{SHAP values}|$ [kWp/hh]")
    
    
    
    # Aesthetics
    fig.tight_layout()

    cax = fig.add_axes([.85, .425, 0.02 , .3])
    cbar_ticks = np.array([1, 4, 7, 10]) 
    cbar = fig.colorbar(sm, cax=cax, ticks=cbar_ticks + .5)
    cbar.set_label('number feature occurs in runs', rotation=270, labelpad=25, size=labelsize,
                   )
    cbar.ax.tick_params(labelsize=labelsize *1.3)
    cbar.ax.tick_params(which='minor', length=0)
    cbar.set_ticklabels(cbar_ticks)

    # Add cartoons
    pv_image = Image.open(f"{__cartoon_path}/pv.png")
    ax_pv_image = fig.add_axes([0.7, .1, 0.25, 0.25])
    ax_pv_image.imshow(pv_image)
    ax_pv_image.axis('off')

    if save_fig:
        fpath_fig = "plots/SI_lasso_shap_feature_importance.pdf"
        fig.savefig(fpath_fig, bbox_inches='tight')

        fig.clear()
        plt.close(fig)

    else:
        plt.show()

    return df_fi


def plot_all_figures():

    use_normalized = True

    # Figure 1: Target distribution on a map of Germany.
    plot_map_distribution_pv_and_bev(save_fig=True)

    # Figure 2: Method overview parts
    plot_separate_parts_method_figure(save_fig=True)

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
    # Figure S1 + S2: All feature dependencies PV
    plot_all_dependencies_separate(feature_count_threshold_pv=15,
                                   feature_count_threshold_bev=15, 
                                   save_fig=True, run_evaluation=False, 
                                   use_normalized=use_normalized)

    # Figure S3: Interaction heatmaps PV and BEV
    plot_interaction_heatmaps_pv_and_bev(save_fig=True, run_evaluation=False,
                                         feature_count_threshold_pv=15, 
                                         feature_count_threshold_bev=15, 
                                         use_normalized=use_normalized)

    # Figure S4: Decomposed SHAP interactions BEV
    plot_decomposed_shap_interactions_pv_and_bev(save_fig=True, target_type='bev',
                                                 run_evaluation=False, 
                                                 feature_count_threshold_pv=15, 
                                                 feature_count_threshold_bev=15,
                                                 use_normalized=use_normalized)
    
    # SI Figure S5: Lasso performance for large alpha values
    plot_performance_large_alpha(save_fig=True)

    # SI Figure S6: Lasso performance and coefficients for small alpha values
    plot_performance_and_coefficients_small_alpha(save_fig=True)

    # SI Figure S7: Lasso SHAP feature importance
    plot_benchmark_shap_feature_importance(save_fig=True)

    return
