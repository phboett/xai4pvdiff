import os
import sys

if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import numpy as np
import pandas as pd
import random

from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold, train_test_split
import shap

from tqdm import tqdm

from datetime import datetime

from copy import deepcopy

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from xai_green_tech_adoption.utils.utils import *
from xai_green_tech_adoption.utils import post_to_mattermost


def random_search_cv(train_set: list, val_set: list, param_intervals: list, n_iter:int, 
                     early_stopping_rounds: int, k_splits: int, verbose: int):
    '''
    Function performs n_iter rounds of random search for hyperparameter optimization (hpo) and returns the best model
    instance and a dataframe containing the hyperparameter values and performances of all rounds. The hyperparameters
    and the intervals to draw them from are given by param_intervals. K-fold cross-validation is performed (with
    k_splits) on data set train_set. The function implements early stopping on the (constant) validation set val_set.
    To determine the number of trees ("n_estimators") for set of hyperparameters, the model is retrained on the entire
    data set train_set while early stopping is performed on the validation set.
    @param train_set: Dataset for k-fold cross-validation. Tuple of dataframe as input and series as target features.
    @param val_set: Dataset for early stopping. Tuple of dataframe as input and series as target features.
    @param param_intervals: Dictionary with hyperparameters used as keys and lists of respective interval bounds as
                            values.
    @param n_iter: Number of rounds of random search
    @param early_stopping_rounds: Number of rounds used for early stopping
    @param k_splits: number of folds of k-fold cross-validation
    @param verbose: verbose parameter used for fitting of LGBMRegressor indicating the performance on validation data
                    in every boosting step

    @return: model instance of the best hyperparameters trained on entire train_set data set and
             dataframe containing the hyperparameter values and performances for all rounds of the random search.
    '''

    X_train, y_train = train_set

    df_performances = pd.DataFrame(columns=[param for param in param_intervals] + [mean_r2_cv_train,mean_r2_cv_test,
                                                                                   mean_mae_cv_train,mean_mae_cv_test,
                                                                                   mean_mape_cv_train,mean_mape_cv_test])

    for iter in range(n_iter):

        kf = KFold(n_splits=k_splits, shuffle=True)

        params = {}
        params['n_estimators'] = param_intervals['n_estimators'][0]
        params['boosting_type'] = param_intervals['boosting_type'][0]
        # draw random parameters
        params['objective'] = random.choice(param_intervals['objective'])
        params['num_leaves'] = np.random.randint(param_intervals['num_leaves'][0], param_intervals['num_leaves'][1] + 1)
        params['min_child_samples'] = np.random.randint(param_intervals['min_child_samples'][0],
                                              param_intervals['min_child_samples'][1] + 1)
        params['learning_rate'] = np.around(np.random.uniform(low=param_intervals['learning_rate'][0],
                                          high=param_intervals['learning_rate'][1]),decimals=3)
        params['subsample'] = np.around(np.random.uniform(low=param_intervals['subsample'][0], high=param_intervals['subsample'][1]),decimals=3)
        params['subsample_freq'] = np.random.randint(param_intervals['subsample_freq'][0],
                                           param_intervals['subsample_freq'][1] + 1)

        r2_train = []
        r2_test = []
        mae_train = []
        mae_test = []
        mape_train = []
        mape_test = []
        # k-fold cross-validation
        for train_idx, test_idx in kf.split(train_set[0]):
            model = LGBMRegressor(**params, n_jobs=-1, verbosity=verbose)
            model.fit(X_train.iloc[train_idx, :], y_train.iloc[train_idx], eval_set=val_set,
                      early_stopping_rounds=early_stopping_rounds, verbose=verbose > 0)
            y_pred_train = model.predict(X_train.iloc[train_idx, :])
            y_pred_test = model.predict(X_train.iloc[test_idx, :])
            r2_train.append(r2_score(y_true=y_train.iloc[train_idx], y_pred=y_pred_train))
            r2_test.append(r2_score(y_true=y_train.iloc[test_idx], y_pred=y_pred_test))
            mae_train.append(mean_absolute_error(y_true=y_train.iloc[train_idx], y_pred=y_pred_train))
            mae_test.append(mean_absolute_error(y_true=y_train.iloc[test_idx], y_pred=y_pred_test))
            mape_train.append(mean_absolute_percentage_error(y_true=y_train.iloc[train_idx], y_pred=y_pred_train))
            mape_test.append(mean_absolute_percentage_error(y_true=y_train.iloc[test_idx], y_pred=y_pred_test))

        # retrain model on entire training set and perform early stopping on the validation set to determine n_estimators
        model = LGBMRegressor(**params, n_jobs=-1, verbosity=verbose)
        model.fit(X_train, y_train, eval_set=val_set, 
                  early_stopping_rounds=early_stopping_rounds, verbose=verbose > 0)
        params.update({
                'n_estimators': model.best_iteration_,
                mean_r2_cv_train: np.mean(r2_train),
                mean_r2_cv_test: np.mean(r2_test),
                mean_mae_cv_train: np.mean(mae_train),
                mean_mae_cv_test: np.mean(mae_test),
                mean_mape_cv_train: np.mean(mape_train),
                mean_mape_cv_test: np.mean(mape_test)})
        df_performances = pd.concat([df_performances, pd.DataFrame([params])], ignore_index=True)

    df_performances[ranking_mean_r2_desc] = df_performances[mean_r2_cv_test].rank(ascending=False)
    best_param = \
    df_performances.loc[df_performances[ranking_mean_r2_desc] == 1, list(param_intervals.keys())].to_dict(
        'records')[0]
    best_model = LGBMRegressor(**best_param, n_jobs=-1, 
                               verbosity=verbose)
    best_model.fit(X_train, y_train, verbose=verbose > 0)

    return best_model, df_performances


def iterate_rfe(train_set_original: tuple[np.ndarray, np.ndarray], 
                val_set_original: tuple[np.ndarray, np.ndarray], 
                param_intervals: dict, n_hpo_iter: int,
                early_stopping_rounds: int, k_splits_cv: int, 
                elimination_scheme: list[int , ...], verbose: int):
    '''
    Function performs Recursive Feature Elimination (RFE). In each iteration it calls random_search_cv() for
    hyperparameter optimization. It eliminates input features according to the scheme given by elimination_scheme and
    returns a dataframe with the hyperparameter values and performances of all runs.
    @param train_set_original: Training set for 5-fold cross-validation performed by random_search_cv().
    @param val_set_original: Validation set for early stopping during hpo performed by random_search_cv().
    @param param_intervals: Dictionary: Keys giving (names of hyperparameters), values give the (endpoints of the)
                            intervals to draw the hyperparameters from.
    @param n_hpo_iter: Number of rounds of random search in each iteration of the rfe.
    @param early_stopping_rounds: Number of rounds used for early stopping during hpo (passed to random_search_cv()).
    @param k_splits_cv: Number of folds of k-fold cross-validation during random search performed by random_search_cv().
    @param elimination_scheme: List of integers indicating how many input features should be eliminated in the
                                iterations simultaneously.
    @param verbose: verbose parameter used during fitting of LGBMRegressor model (by random_search_cv())

    @return: Dataframe giving hyperparameter values and performances of all runs during hpo for all iterations of rfe,
                i.e., the dataframe contains (len(elimination_scheme)+1)*n_hpo_iter rows.
    '''
    train_set = deepcopy(train_set_original)
    val_set = deepcopy(val_set_original)

    best_model, df_perf = random_search_cv(train_set=train_set,
                                           val_set=val_set,
                                           param_intervals=param_intervals,
                                           n_iter=n_hpo_iter,
                                           early_stopping_rounds=early_stopping_rounds,
                                           k_splits=k_splits_cv,
                                           verbose=verbose)

    df_perf[col_feature_count] = train_set[0].shape[1]
    df_perf[col_features] = df_perf.shape[0] * [list(train_set[0].columns)]

    for count_feat_to_elim in elimination_scheme:
        # determine shap values
        shap_expl = shap.TreeExplainer(best_model)
        shap_values_val = shap_expl(val_set[0])
        df_shap_vals = pd.DataFrame(shap_values_val.values, columns=val_set[0].columns)
        mean_shap_val = df_shap_vals.abs().mean(axis=0)
        # drop the least important features
        features_to_keep = mean_shap_val.sort_values(ascending=True)[count_feat_to_elim:].keys()
        train_set = (train_set[0][features_to_keep], train_set[1])
        val_set = (val_set[0][features_to_keep], val_set[1])
        # perform hpo for reduced model
        best_model, df_perf_reduced = random_search_cv(train_set=train_set,
                                                       val_set=val_set,
                                                       param_intervals=param_intervals,
                                                       n_iter=n_hpo_iter,
                                                       early_stopping_rounds=early_stopping_rounds,
                                                       k_splits=k_splits_cv,
                                                       verbose=verbose)
        # add number of features and list of features considered by model
        df_perf_reduced[col_feature_count] = train_set[0].shape[1]
        df_perf_reduced[col_features] = df_perf_reduced.shape[0] * [list(train_set[0].columns)]

        df_perf = pd.concat([df_perf, df_perf_reduced], ignore_index=True)

    return df_perf


def repeat_rfe(file_path: str, test_size: float, validation_size: float, 
               repetitions: int, param_intervals: dict, n_hpo_iter: int,
               early_stopping_rounds: int, k_splits_cv: int, 
               elimination_scheme: list, label_cols: list, 
               target_feat, verbose, show_progress: bool = True,
               norm_ls: list = [], drop_ls: list = []):
    '''
    Entire simulation run of repeated RFE on repetitions different train-test-splits of the dataset. For each choice of
    train-test-splitting, the algorithm performs RFE.
    @param file_path: Path to input data
    @param test_size: proportion of data which is used for testing
    @param validation_size: proportion of the REMAINING data (i.e., the training set of size (1-test_size)) which is
            used for validation
    @param repetitions: number of different simulation runs, i.e., different training-test-splits
    @param param_intervals: intervals used for HPO
    @param n_hpo_iter: iterations of random search in each iteration of RFE
    @param early_stopping_rounds: number of early stopping round
    @param k_splits_cv: number of splits used for K-fold cross-validation during hyperparameter optimisation
    @param elimination_scheme: scheme indicating numbers of input features that are eliminated in each round of RFE
    @param label_cols: names of columns giving labels, i.e., name and RS of municipal associations
    @param target_feat: column name of the column giving the target feature
    @param verbose: verbose parameter used during fitting of LGBMRegressor model
    @param show_progress: boolean indicating whether progress bar should be shown.
    @param norm_ls: list of features to be normalized to population.
    @param drop_ls: list of features to be dropped from the dataset.

    @return: df_perf: dataframe giving the results and performances of the simulation. One row gives the details for one
                        set of hyperparameters for a specific number of input features for a specific repetition. (The
                        dataframe has repetitions*(len(elimination scheme)+1)*n_hpo_iter entries.)
            df_metadata: Dataframe giving the metadata of the simulation.
    '''

    data = pd.read_csv(file_path, sep=';')

    if len(drop_ls) > 0:
        data = data.drop(columns=drop_ls)

    if len(norm_ls) > 0:    
        for feat_r in norm_ls:
            feat_new = feat_r + '_per_capita'
            data[feat_new] = data[feat_r] / data['population'].astype(float)
            data.drop(columns=feat_r, 
                        inplace=True) 

    y = data[target_feat]
    X = data.drop([target_feat] + label_cols, axis=1)

    assert sum(elimination_scheme) + 1 == len(X.columns), 'Elimination scheme suggests another number of features.'

    for rep in tqdm(range(repetitions), 
                    desc='Repetitions', 
                    disable=not show_progress):
        [X_train, X_test, 
         y_train, y_test] = train_test_split(X, y, 
                                             test_size=test_size)
        [X_train, X_val, 
         y_train, y_val] = train_test_split(X_train, y_train, 
                                            test_size=validation_size)
        idx_train = list(X_train.index)
        idx_val = list(X_val.index)
        idx_test = list(X_test.index)
        train_set = (X_train, y_train)
        val_set = (X_val, y_val)

        # perform recursive feature elimination
        df_perf_run = iterate_rfe(train_set_original=train_set,
                                  val_set_original=val_set,
                                  param_intervals=param_intervals,
                                  n_hpo_iter=n_hpo_iter,
                                  early_stopping_rounds=early_stopping_rounds,
                                  k_splits_cv=k_splits_cv,
                                  elimination_scheme=elimination_scheme,
                                  verbose = verbose)
        df_perf_run['run'] = rep

        df_metadata_run = pd.DataFrame({
                                    col_file_path: file_path,
                                    col_target_feat: target_feat,
                                    'validation_size': validation_size * (1 - test_size),
                                    'test_size': test_size,
                                    'repetitions_of_rfe': repetitions,
                                    'parameter_intervals': [param_intervals],
                                    'hpo_iterations': n_hpo_iter,
                                    'early_stopping_rounds': early_stopping_rounds,
                                    'k_splits_cv': k_splits_cv,
                                    'elimination_scheme': [elimination_scheme],
                                    col_idx_train: [idx_train],
                                    col_idx_val: [idx_val],
                                    col_idx_test: [idx_test],
                                    col_run_id: rep})
        if rep == 0:
            df_perf = df_perf_run.copy()
            df_metadata = df_metadata_run.copy()
        else:
            df_perf = pd.concat([df_perf, df_perf_run], ignore_index=True)
            df_metadata = pd.concat([df_metadata,df_metadata_run],ignore_index=True)


    return df_perf, df_metadata


if __name__ == '__main__':

    t_start = datetime.now()
    
    number_of_input_arguments = len(sys.argv) - 1
    if number_of_input_arguments != 2 or sys.argv[1] not in ['pv', 'bev']:
        raise IOError('Please choose the type of target (i.e., {pv, bev}) to be run.')    
    
    target_type = sys.argv[1]
    if abs(float(sys.argv[2])) < 1e-3:
        norm_ls_in = []
        drop_ls_in = []
    else:
        norm_ls_in = features_norm_to_population_ls
        drop_ls_in = features_norm_drop_ls

        if target_type == 'bev':
            norm_ls_in += features_norm_to_population_bev_ls

    print(f"Starting RFE for target type: {target_type}")

    # set random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # number of repetitions of entire workflow --> robustness check
    rep = 10

    # fraction of data set considered as test set
    test_size = 0.2
    # fraction of the remaining data set considered as validation set
    validation_size = 0.2

    # parameter intervals for random search for hyperparameter optimization (hpo)
    param_intervals_endpoints = {
        'objective': ['l2'],
        'n_estimators': [500],
        'num_leaves': [2, 10],
        'min_child_samples': [40, 200],
        'learning_rate': [0.001, 0.1],
        'subsample': [0.5, 1],
        'subsample_freq': [0, 5],
        'boosting_type': ['gbdt']
    }
    # number of iterations of random search for hpo
    n_iter = 50

    # number of splits for k-fold cross-validation applied during random search for hpo
    k_splits = 5
    n_jobs = -1

    early_stopping_rounds = 30
    verbose = -1

    # scheme to eliminate features during recursive feature elimination
    if target_type == 'pv':
        list_feat_to_elim = 11 * [10] + 9 * [5] + 12 * [2] + (18 - len(drop_ls_in)) * [1]
        # file path of input data set
        file_path = 'data/input/input.csv'
        # file path on cluster
        col_target = col_power_accum_pv

    elif target_type == 'bev':
        list_feat_to_elim = 11 * [10] + 9 * [5] + 12 * [2] + (19 - len(drop_ls_in)) * [1] #to get to 15 features
        # file path of input data set
        file_path = 'data/input/bev_input.csv'
        # file path on cluster
        col_target = col_bev_per_vehicle

    t_start = datetime.now()

    df_perf, df_metadata = repeat_rfe(file_path, validation_size, test_size, repetitions=rep,
                                      param_intervals=param_intervals_endpoints, n_hpo_iter=n_iter,
                                      early_stopping_rounds=early_stopping_rounds, k_splits_cv=k_splits,
                                      elimination_scheme=list_feat_to_elim,
                                      label_cols=[col_id_ma, col_name_ma], 
                                      target_feat=col_target,verbose=verbose,
                                      norm_ls=norm_ls_in, drop_ls=drop_ls_in)
    
    # file path on cluster
    out_path = 'data/output/'
    perf_path = out_path + 'results_rfe'
    meta_path = out_path + 'metadata_rfe'

    bev_path_tmp = ''
    if target_type == 'bev':
        bev_path_tmp = '_bev'

    norm_path_tmp = ''
    if len(norm_ls_in) > 0:
        norm_path_tmp = '_norm'


    df_perf.to_csv(perf_path + bev_path_tmp + norm_path_tmp + '.csv', 
                   index=False, sep=';')
    df_metadata.to_csv(meta_path + bev_path_tmp + norm_path_tmp + '.csv', 
                       index=False, sep=';')
    if mattermost_url is not None:
        dtime = (t_start - datetime.now()).total_seconds() / 3600.
        message_to_post = 'RFE for ' + target_type + f' finished at {datetime.now()} (took {dtime} h)'
        post_to_mattermost.post_message(message_to_post, mattermost_url)