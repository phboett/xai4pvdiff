#!/usr/bin/env python
# -*- coding: utf-8 -*

"""Plot for producing the figures for the paper 
that presents analysis of the diffusion of both pv and bev."""

import os
import sys

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from PIL import Image

import shap

import pickle
import gzip

__fpath_output = "data/output"
__cartoon_path = "data/raw_data/plotting"

sys.path.append("code")
from utils.utils import col_feature_count, ranking_mean_r2_desc, \
    col_run_id, mean_r2_cv_test, mean_r2_cv_train, col_file_path, \
    col_bev_per_vehicle, col_power_accum_pv, col_features, \
    col_idx_train, col_idx_val, col_idx_test, mean_r2_cv_test, col_mean_shap

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
                        'CDU/CSU': 'votes CDU/CSU [%]',
                        'other parties': 'votes other parties [%]',
                        'FDP': 'votes FDP [%]',
                        'The Left': 'votes The Left [%]',
                        'AfD': 'votes AfD [%]',
                        'The Greens': 'votes The Greens [%]',
                        'SPD': 'votes SPD [%]',
                        'share 4-person households': 'share of\n4-person households [%]',
                        'share 5-person households': 'share of\n5-person households [%]',
                        'flats with 5+ rooms': 'share of\nflats with 5+ rooms [%]',
                        col_power_accum_pv: 'photovoltaic power [kW p.h.]',
                        'global radiation': 'global radiation [kWh/m²]',
                        'income tax': 'income tax [€ p.c.]',}

def load_performance_and_metadata_dataframes():

    df_perf_pv = pd.read_csv(f"{__fpath_output}/results_rfe.csv", 
                             sep=';')
    df_perf_pv = prepare_performance_dataframe(df_perf_pv, 'features')

    df_perf_bev = pd.read_csv(f"{__fpath_output}/results_rfe_bev.csv", 
                              sep=';')
    df_perf_bev = prepare_performance_dataframe(df_perf_bev, 'features')

    
    df_metadata_pv = pd.read_csv(f"{__fpath_output}/metadata_rfe.csv",
                                    sep=';')
    df_metadata_pv = prepare_metadata_dataframe(df_metadata_pv, 
                                                ["indices_training_set", 
                  "indices_val_set", 
                  "indices_test_set"])  

    df_metadata_bev = pd.read_csv(f"{__fpath_output}/metadata_rfe_bev.csv",
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


def get_details_about_best_model(df_perf, df_metadata, 
                                 model_dict,
                                 split_id_best_model,
                                 feature_count_threshold):
    
    df_data = pd.read_csv(df_metadata[col_file_path].unique()[0], sep=";")

    # TODO delete
    """idx_train = df_metadata.loc[df_metadata[col_run_id] == split_id_best_model, 
                                col_idx_train].values[0]
    idx_val = df_metadata.loc[df_metadata[col_run_id] == split_id_best_model, 
                              col_idx_val].values[0]
    idx_test = df_metadata.loc[df_metadata[col_run_id] == split_id_best_model, 
                               col_idx_test].values[0]"""

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

def plot_map_distribution_pv_and_bev(save_fig=False):
    """Plot the final"""

    # Plot
    fig, [ax_pv, ax_bev] = plt.subplots(1, 2, figsize=(12, 6))


    # Load dataset

    # Load dictionary connecting ars to wkt coordinates

    ## Plot PV


    ## Plot BEV


    # Aesthetics


    if save_fig:
        fig_path = "plots/map_distribution_pv_and_bev.pdf"
        fig.savefig(fig_path, bbox_inches='tight')

        fig.clear()
        plt.close(fig)

    else:
        plt.show()

    return 


def plot_recursive_gbt_performance(feature_count_threshold_pv=15, 
                                   feature_count_threshold_bev=16,
                                   save_fig=False):

    [df_perf_pv, df_perf_bev, 
     df_metadata_pv, df_metadata_bev] = load_performance_and_metadata_dataframes()
    
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


def plot_mean_shap_features(df_mean_shap_pv,
                             df_mean_shap_bev,  
                             save_fig=False):
    
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


def plot_dependency_pv_and_bev(feature_count_threshold_pv=15,
                              feature_count_threshold_bev=16, 
                              save_fig=False, nr_best_features=4,
                              run_evaluation=False):
    """Plot 2 by 4 with depency plot in the best model."""

    fpath_plot_data = "data/intermediate_data/dependency_plots_pv_and_bev.pklz"

    if not os.path.exists(fpath_plot_data) or run_evaluation: 
        [df_perf_pv, df_perf_bev, 
        df_metadata_pv, df_metadata_bev] = load_performance_and_metadata_dataframes()
        
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
        shap_values_pv, _] = get_details_about_best_model(df_perf_pv, df_metadata_pv, model_dict_pv, 
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
        shap_values_bev, _] = get_details_about_best_model(df_perf_bev, df_metadata_bev, model_dict_bev, 
                                                                                    split_id_best_model, feature_count_threshold_bev)

        res_tup = (X_red_model_pv, shap_values_pv, df_mean_shap_pv,
                   X_red_model_bev, shap_values_bev, df_mean_shap_bev)
        
        with gzip.open(fpath_plot_data, "wb") as fh_plot_data:
            pickle.dump(res_tup, fh_plot_data)
    
    else:
        with gzip.open(fpath_plot_data, "rb") as fh_plot_data:
            (X_red_model_pv, shap_values_pv, df_mean_shap_pv,
            X_red_model_bev, shap_values_bev, df_mean_shap_bev) = pickle.load(fh_plot_data)


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
                                   feature_count_threshold_bev=16,
                                   save_fig=False, run_evaluation=False,
                                   labelsize=14):
    """Plot shap values over the feature values for all features of the best number."""

    fpath_plot_data = "data/intermediate_data/dependency_plots_pv_and_bev.pklz"

    if not os.path.exists(fpath_plot_data) or run_evaluation: 
        [df_perf_pv, df_perf_bev, 
        df_metadata_pv, df_metadata_bev] = load_performance_and_metadata_dataframes()
        
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
        shap_values_pv, _] = get_details_about_best_model(df_perf_pv, df_metadata_pv, model_dict_pv, 
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
        shap_values_bev, _] = get_details_about_best_model(df_perf_bev, df_metadata_bev, model_dict_bev, 
                                                                                    split_id_best_model, feature_count_threshold_bev)

        res_tup = (X_red_model_pv, shap_values_pv, df_mean_shap_pv,
                   X_red_model_bev, shap_values_bev, df_mean_shap_bev)
        
        with gzip.open(fpath_plot_data, "wb") as fh_plot_data:
            pickle.dump(res_tup, fh_plot_data)
    
    else:
        with gzip.open(fpath_plot_data, "rb") as fh_plot_data:
            (X_red_model_pv, shap_values_pv, df_mean_shap_pv,
            X_red_model_bev, shap_values_bev, df_mean_shap_bev) = pickle.load(fh_plot_data)

    # Plot
    ## PV
    mean_shap_red_pv = np.mean(abs(shap_values_pv), axis=0)
    idx_mean_shap_sort_pv = np.argsort(mean_shap_red_pv)[::-1]
    fig_cols_pv = len(mean_shap_red_pv) // 4
    remainder_pv = len(mean_shap_red_pv) % 4
    if remainder_pv > 0:
        fig_cols_pv += 1
    fig_pv, ax_arr_pv = plt.subplots(fig_cols_pv, 4, 
                                 figsize=(20, 5 * fig_cols_pv),                    
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
    fig_cols_bev = len(mean_shap_red_bev) // 4
    remainder_bev = len(mean_shap_red_bev) % 4
    if remainder_bev > 0:
        fig_cols_bev += 1
    fig_bev, ax_arr_bev = plt.subplots(fig_cols_bev, 4, 
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
    fig_pv.subplots_adjust(hspace=.4, right=0.925)
    fig_pv.canvas.manager.set_window_title("Feature depence plots for PV")

    fig_bev.tight_layout()
    fig_bev.subplots_adjust(hspace=.4, right=0.925)
    fig_bev.canvas.manager.set_window_title("Feature depence plots for BEV")

    # Add cartoons
    pv_image = Image.open(f"{__cartoon_path}/pv.png")
    ax_pv_image = fig_pv.add_axes([0.875, .825, 0.17, 0.17])
    ax_pv_image.imshow(pv_image)
    ax_pv_image.axis('off')


    car_image = Image.open(f"{__cartoon_path}/car.png")
    ax_car_image = fig_bev.add_axes([0.89, .9, 0.1, 0.1])
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
