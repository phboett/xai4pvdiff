import sys
import os

if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, \
    mean_absolute_error, mean_absolute_percentage_error
import random

from tqdm import tqdm

from xai_green_tech_adoption.utils.utils import *
from xai_green_tech_adoption.utils import post_to_mattermost
from xai_green_tech_adoption.shap_analysis.supporting_functions import *

from datetime import datetime


def lasso_simulation(df: pd.DataFrame, col_target_feature: str, 
                     train_sets: dict, test_sets: dict, 
                     alpha_range: list, 
                     show_progress: bool = True, max_iter: int = 20000):
    '''
    For each value of alpha contained in the list alpha_range a lasso model is trained and tested.
    @param df: dataset of input features and target feature
    @param col_target_feature: column name of column with target feature
    @param train_sets: dictionary with indices of training set (joint training and validation set from GBT simulation)
                        for each split
    @param test_sets: dictionary with indices of test set (joint training and validation set from GBT simulation) for
                        each split
    @param alpha_range: list of alpha values. A lasso model is created, trained and tested for each value of alpha.
    @param show_progress: if True, progress bar is shown
    @param max_iter: maximum number of allowed iterations for lasso model

    @return: Dataframe with number of training-test splits, value of alpha, coefficients for all input features and values of
                performance evaluation metrics
    '''
    
    y_pv = df[col_target_feature]
    X = df.drop([col_id_ma, col_name_ma, col_target_feature], axis=1)

    df_lasso_perf = pd.DataFrame()
    for idx_run, run in tqdm(enumerate(train_sets), 
                             disable=not show_progress, desc="Runs", 
                             total=len(train_sets)):
        X_train = X.iloc[train_sets[run], :]
        X_test = X.iloc[test_sets[run], :]
        y_train = y_pv.iloc[train_sets[run]]
        y_test = y_pv.iloc[test_sets[run]]

        # standardize data
        std_scaler = StandardScaler()
        std_scaler.fit(X_train)
        X_train_scaled = std_scaler.transform(X_train)
        X_test_scaled = std_scaler.transform(X_test)

        for alpha in alpha_range:
            lasso = Lasso(alpha=alpha, tol=0.001, max_iter=max_iter)
            lasso.fit(X_train_scaled, y_train)
            y_train_pred = lasso.predict(X_train_scaled)
            y_test_pred = lasso.predict(X_test_scaled)
            feature_names = list(X_train.columns)
            coef = lasso.coef_
            coef_dict = {feature: coef for feature, coef in zip(feature_names, coef)}
            df_alpha = pd.DataFrame([{col_run_id: run,
                                      col_alpha: alpha,
                                      col_r2_train: r2_score(y_true=y_train, y_pred=y_train_pred),
                                      col_r2_test: r2_score(y_true=y_test, y_pred=y_test_pred),
                                      col_l2_train: np.linalg.norm(y_train - y_train_pred, ord=2) ** 2,
                                      col_l2_test: np.linalg.norm(y_test - y_test_pred, ord=2) ** 2,
                                      col_mse_train: mean_squared_error(y_true=y_train, y_pred=y_train_pred),
                                      col_mse_test: mean_squared_error(y_true=y_test, y_pred=y_test_pred),
                                      col_mae_train: mean_absolute_error(y_true=y_train, y_pred=y_train_pred),
                                      col_mae_test: mean_absolute_error(y_true=y_test, y_pred=y_test_pred),
                                      col_mape_train: mean_absolute_percentage_error(y_true=y_train,
                                                                                     y_pred=y_train_pred),
                                      col_mape_test: mean_absolute_percentage_error(y_true=y_test, y_pred=y_test_pred),
                                      **coef_dict}])
            df_lasso_perf = pd.concat([df_lasso_perf, df_alpha], ignore_index=True)

    return df_lasso_perf


if __name__ == '__main__':

    t_start = datetime.now()
    number_of_input_arguments = len(sys.argv) - 1
    if number_of_input_arguments != 2 or sys.argv[1] not in ['pv', 'bev']:
        raise IOError('Please choose the type of target ' + 
                      '(i.e., {pv, bev}) to be run.')    
    
    target_type = sys.argv[1]
    if abs(float(sys.argv[2])) < 1e-3:
        norm_ls = []
        drop_ls = []
    else:
        norm_ls = features_norm_to_population_ls
        drop_ls = features_norm_drop_ls

        if target_type == 'bev':
            norm_ls += features_norm_to_population_bev_ls

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    pd.options.display.float_format = '{:20,.17f}'.format

    
    fpath_gbt_results = 'data/output/metadata_rfe'
    if target_type == 'bev':
        fpath_gbt_results += '_bev'

    if len(norm_ls) > 0:
        fpath_gbt_results += '_norm'

    df_metadata_gbt = pd.read_csv(fpath_gbt_results + '.csv', sep=';')
    df_metadata_gbt = prepare_metadata_dataframe(df_metadata=df_metadata_gbt,
                                                 idx_sets=[col_idx_train, col_idx_val, col_idx_test])
    df_input = pd.read_csv(df_metadata_gbt[col_file_path].unique()[0], sep=';')
    
    if len(norm_ls) > 0 and len(drop_ls) > 0:
        
        df_input = df_input.drop(columns=drop_ls)

        for feat_r in norm_ls:
            feat_new = feat_r + '_per_capita'
            df_input[feat_new] = df_input[feat_r] / df_input['population'].astype(float)
            df_input.drop(columns=feat_r, 
                          inplace=True)    

    target_feature = df_metadata_gbt[col_target_feat].values[0]

    y_pv = df_input[target_feature]
    labels = df_input[[col_name_ma, col_id_ma]]

    # change infinity values (of feature 'mean distance public transport') to max value as lasso cannot handle
    # inf values
    df_input.loc[df_input.isin([np.inf, -np.inf]).any(axis=1), 'mean distance public transport'] = \
        df_input.loc[~df_input.isin([np.inf, -np.inf]).any(axis=1), 'mean distance public transport'].max()

    # choose values for alpha
    alpha_range_large = 1. / np.linspace(1, 10000, 
                                         num=1000, endpoint=True)
    alpha_range = 1. / np.linspace(0.001, 50, 
                                   num=1000, endpoint=True)

    train_indices = {run: df_metadata_gbt.loc[df_metadata_gbt[col_run_id] == run, 
                                              col_idx_train].values[0] +
                          df_metadata_gbt.loc[df_metadata_gbt[col_run_id] == run, 
                                              col_idx_val].values[0] for run in
                     df_metadata_gbt[col_run_id].unique()}
    test_indices = {run: df_metadata_gbt.loc[df_metadata_gbt[col_run_id] == run, 
                                             col_idx_test].values[0] for run in
                    df_metadata_gbt[col_run_id].unique()}

    # simulation for small values of alpha, i.e., large values of inverse alpha
    df_lasso_perf_large = lasso_simulation(df=df_input, col_target_feature=target_feature, train_sets=train_indices,
                                           test_sets=test_indices, alpha_range=alpha_range_large)
    # simulation for large values of alpha, i.e., small values of inverse alpha
    df_lasso_perf = lasso_simulation(df=df_input, col_target_feature=target_feature, train_sets=train_indices,
                                     test_sets=test_indices, alpha_range=alpha_range)

    fpath_out_dir = 'data/output'
    fpath_lasso_test = fpath_out_dir + '/benchmarking_lasso'
    fpath_lasso_large_alpha_test = fpath_out_dir + '/benchmarking_lasso_large_alpha'
    if target_type == 'bev':
        fpath_lasso_test += '_bev'
        fpath_lasso_large_alpha_test += '_bev'

    if len(norm_ls) > 0:
        fpath_lasso_test += '_norm'
        fpath_lasso_large_alpha_test += '_norm'

    """df_lasso_perf.to_csv(fpath_lasso_test + '.csv', sep=';', index=False,
                         float_format='{:f}'.format)
    df_lasso_perf_large.to_csv(fpath_lasso_large_alpha_test + '.csv',
                               sep=';', index=False, float_format='{:f}'.format)"""
    
    # output as pickled object
    df_lasso_perf.to_pickle(fpath_lasso_test + '.pklz', compression='gzip')
    df_lasso_perf_large.to_pickle(fpath_lasso_large_alpha_test + '.pklz', compression='gzip')

    if mattermost_url is not None:
        dtime = (t_start - datetime.now()).total_seconds() / 3600
        message_to_post = 'Benchmarking ' + target_type + f' finished at {datetime.now()} (took {dtime} h)'
        post_to_mattermost.post_message(message_to_post, mattermost_url)
