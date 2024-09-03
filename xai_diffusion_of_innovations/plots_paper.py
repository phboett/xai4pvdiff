#!/usr/bin/env python
# -*- coding: utf-8 -*

import sys

import pandas as pd

from matplotlib import pyplot as plt
from PIL import Image

__fpath_output = "data/output"
__cartoon_path = "data/raw_data/plotting"

sys.path.append("code")
from utils.utils import col_feature_count, ranking_mean_r2_desc, \
    col_run_id, mean_r2_cv_test, mean_r2_cv_train, col_file_path, \
    col_bev_per_vehicle, col_power_accum_pv, col_features

sys.path.append("code/shap_analysis")
import plotting
from supporting_functions import prepare_performance_dataframe, \
    prepare_metadata_dataframe, get_fitted_models as __get_fitted_models, \
    get_mean_shap as __get_mean_shap


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
        train_on_train_val=True)
    
    df_mean_shap = __get_mean_shap(
        model_dict,
        feature_count_threshold=feature_count_threshold,
        df_perf=df_perf,
        df_metadata=df_metadata)

    return model_dict, df_mean_shap


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
    
    df_perf_all_split_bev = df_perf_bev.loc[
    (df_perf_bev[col_feature_count] == feature_count_threshold_bev)
    & (df_perf_bev[ranking_mean_r2_desc] == 1),
    [col_run_id, mean_r2_cv_test],
    ]
    split_id_best_model = df_perf_all_split_bev.loc[
        df_perf_all_split_bev[mean_r2_cv_test] == df_perf_all_split_bev[mean_r2_cv_test].max(),
        col_run_id,
    ].values[0]
    print("The id of train-test split with best performance for BEV: ", 
          split_id_best_model)

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
        fig_path = "rfe_performance_pv_and_bev.pdf"
        fig.savefig(fig_path, bbox_inches='tight')
    else:
        plt.show()

    return


def plot_mean_shape_features(df_mean_shap_pv,
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
        fig_path = "shap_feature_importance_pv_n_bev.pdf"
        fig.savefig(fig_path, bbox_inches='tight')
    else:
        plt.show()
    
    return

def get_details_about_best_model(df_perf, df_metadata, 
                                 model_dict, df_mean_shap,
                                 split_id_best_model,
                                 feature_count_threshold):
    
    df_data = 

    idx_train = df_metadata.loc[df_metadata[col_run_id] == split_id_best_model, 
                                col_idx_train].values[0]
    idx_val = df_metadata.loc[df_metadata[col_run_id] == split_id_best_model, 
                              col_idx_val].values[0]
    idx_test = df_metadata.loc[df_metadata[col_run_id] == split_id_best_model, 
                               col_idx_test].values[0]

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

def plot_dependency_pv_and_bv(df_mean_shap_pv,
                              df_mean_shap_bev,
                              feature_count_threshold_pv=15,
                              feature_count_threshold_bev=16):
    """Plot 2 by 4 with depency plot in the best model."""

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
    
    df_data_pv = pd.read_csv(df_metadata_pv[col_file_path].unique()[0], sep=";")


    df_performance_red_model = df_perf_pv.loc[
    (df_perf_pv[col_run_id] == split_id_best_model)
    & (df_perf_pv[col_feature_count] == feature_count_threshold_pv)
    & (df_perf_pv[ranking_mean_r2_desc] == 1),
    :,]

    model_dict_pv, df_mean_shap_pv = get_fitted_models_and_mean_shap_values(df_perf_pv, df_metadata_pv,feature_count_threshold_pv, 
                                           col_bev_per_vehicle)
    X_red_model_pv = df_data_pv[df_performance_red_model[col_features].values[0]]
    red_model_pv = model_dict_pv[split_id_best_model]

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


    

    # Make list of best performing features

    # Plot
    fig, [ax_pv, ax_bev] = fig.subplots(2, 4, figsize=(20, 10))

    ## PV row
    plotting.dependency_plot(X_red_model_pv, shap_values_pv,
                             feature="income tax", x_label="income tax",
                             ylabel="SHAP value", ax=ax_pv[0])

    ## BEV row