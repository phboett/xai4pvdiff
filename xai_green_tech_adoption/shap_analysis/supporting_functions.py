#!/usr/bin/env python
# -*- coding: utf-8 -*

import re
import numpy as np
import pandas as pd

import lightgbm
import shap

from tqdm import tqdm

from xai_green_tech_adoption.utils.utils import *

def prepare_performance_dataframe(df_perf, col_features):
    '''Prepare dataframe with results. String of all features to list of features.
    @param df_perf: Dataframe giving results and performances of GBT simulation
    @param col_features: column name of column giving list of features
    @return Dataframe with transformed column of feature list'''

    df_perf[col_features] = df_perf[col_features].apply(lambda l: re.findall(r"'(.*?)'", l))

    return df_perf


def prepare_metadata_dataframe(df_metadata, idx_sets):
    '''Transform dataframe giving metadata of GBT simulation. Transform string of all indices on training, validation
    and test sets to s of indices.
    @param df_metadata: Dataframe giving metadata of GBT simulation
    @param idx_sets: column names of columns giving indices
    @return Transformed dataframe'''
    for indices in idx_sets:
        df_metadata[indices] = df_metadata[indices].apply(
            lambda idx_list: [int(idx) for idx in re.findall(r'\d+', idx_list)])
    return df_metadata


def get_fitted_models(df_perf, df_metadata, target_feat, 
                      feat_count_red_model=None,
                      train_on_train_val=True,
                      show_progress=False, use_normalized: bool = True):
    '''
    Return a dictionary of best reduced models with 'feat_count_red_model' features for all runs. Models are fitted on
    the training (and validation) data sets of the respective run.
    @param df_perf: Dataframe giving results of GBT simulation
    @param df_metadata: Dataframe giving metadata of GBT simulation runs
    @param feat_count_red_model: number of features included in reduced model
    @param target_feat: column anme of target feature
    @param train_on_train_val: Boolean indicating whether the models should be trained on the joint training and
            validation set (default: true)
    @param save_models_to_path: File path to save the resulting dataframe as csv file (optional)
    @param show_progress: Boolean indicating whether the progress bar should be shown (default: false)
    @param use_normalized: Boolean indicating whether the normalized features (e.g., some by employment) should be used (default: True)

    @return: Dictionary of reduced models of all training-test-splits. The numbers of the splittings serve as keys.
    '''

    df_data = pd.read_csv(df_metadata[col_file_path].unique()[0], sep=';')

    if use_normalized:
        norm_ls = features_norm_to_population_ls
        drop_ls = features_norm_drop_ls

        df_data = df_data.drop(columns=drop_ls)
        for feat_r in norm_ls:
            feat_new = feat_r + '_per_capita'
            df_data[feat_new] = df_data[feat_r] / df_data['population'].astype(float)
            df_data.drop(columns=feat_r, 
                         inplace=True) 


    y = df_data[target_feat]
    X = df_data.drop([target_feat, col_id_ma, col_name_ma], axis=1)

    
    model_dict = {}

    for run in tqdm(df_perf[col_run_id].unique(), disable=not show_progress, 
                    desc="Fitting models"):
        if feat_count_red_model != None:
            df_perf_run = df_perf[(df_perf[col_run_id] == run) & (df_perf[ranking_mean_r2_desc] == 1) & (
                    df_perf[col_feature_count] == feat_count_red_model)]
        else:
            raise ValueError('Neither feature count nor r2 threshold are given as input.')
        train_idx = df_metadata.loc[df_metadata[col_run_id] == run, col_idx_train].values[0]
        val_idx = df_metadata.loc[df_metadata[col_run_id] == run, col_idx_val].values[0]

        features = df_perf_run[col_features].values[0]
        param_keys = ['objective', 'n_estimators', 'num_leaves', 'min_child_samples', 'learning_rate',
                      'subsample', 'subsample_freq', 'boosting_type']
        model = lightgbm.LGBMRegressor(**df_perf_run[param_keys].to_dict('records')[0], 
                                       n_jobs=-1)
        if train_on_train_val:
            model.fit(X=X.iloc[train_idx + val_idx, :][features], y=y.iloc[train_idx + val_idx])
        else:
            model.fit(X=X.iloc[train_idx, :][features], y=y.iloc[train_idx])
        model_dict[run] = model
    return model_dict


def get_mean_shap(model_dict, df_perf, 
                  df_metadata, feature_count_threshold,
                  use_normalized: bool = True):
    '''
    Get dataframe with SHAP feature importances.
    @param model_dict: dictionary of reduced model of all ten training-test-splits
    @param df_perf: dataframe giving all performances of the simulation
    @param df_metadata: dtaaframe giving metadta of GBT simulation
    @param feature_count_threshold: number of features included in the reduced model
    @param r2_threshold: Alternatively choose an R2 threshold to determine the reduced models
    @return: Dataframe with feature importances of reduced models of all 10 training-test-splittings.

    '''

    df_data = pd.read_csv(df_metadata[col_file_path].unique()[0], sep=';')

    if use_normalized:
        norm_ls = features_norm_to_population_ls
        drop_ls = features_norm_drop_ls

        df_data = df_data.drop(columns=drop_ls)
        for feat_r in norm_ls:
            feat_new = feat_r + '_per_capita'
            df_data[feat_new] = (df_data[feat_r] / 
                                 df_data['population'].astype(float))
            df_data.drop(columns=feat_r, 
                         inplace=True) 


    list_all_feat = []
    df_mean_shap = pd.DataFrame(columns=model_dict.keys())
    for run in df_perf[col_run_id].unique():
        
        df_perf_run = df_perf.loc[
            (df_perf[col_run_id] == run) & (df_perf[col_feature_count] == feature_count_threshold) &
            (df_perf[ranking_mean_r2_desc] == 1)]

        model = model_dict[run]
        explainer = shap.explainers.Tree(model)
        shap_values = explainer.shap_values(df_data[df_perf_run[col_features].values[0]])
        # getting mean abs of SHAP values for all features
        mean_shap = np.abs(shap_values).mean(axis=0)
        for idx, feature in enumerate(df_data[df_perf_run[col_features].values[0]]):
            df_mean_shap.loc[feature, run] = mean_shap[idx]
        list_all_feat += df_perf_run[col_features].values[0]
    # listing all features occurring in all top-10-runs uniquely
    list_all_feat = [*set(list_all_feat)]
    df_mean_shap[col_occurences_feat] = df_mean_shap.loc[list_all_feat, df_perf[col_run_id].unique()].notna().sum(
        axis=1)
    # if features not contained in reduces model of run: mean SHAP = NaN??
    df_mean_shap.loc[list_all_feat, col_mean_shap] = df_mean_shap.apply(
        lambda row: np.abs(row[df_perf[col_run_id].unique()]).mean(), axis=1)
    df_mean_shap.loc[list_all_feat, col_std_shap] = df_mean_shap.apply(
        lambda row: np.std(np.abs(row[df_perf[col_run_id].unique()])), axis=1)
    df_mean_shap = df_mean_shap.iloc[df_mean_shap[col_mean_shap].argsort(), :]
    for run in df_perf[col_run_id].unique():
        df_mean_shap.loc[mean_r2_cv_test, run] = df_perf.loc[
            (df_perf[col_run_id] == run) & (df_perf[col_feature_count] == feature_count_threshold) &
            (df_perf[ranking_mean_r2_desc] == 1),mean_r2_cv_test].values[0]
        
    return df_mean_shap