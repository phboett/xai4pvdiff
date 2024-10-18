#!/usr/bin/env python
# -*- coding: utf-8 -*

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib import ticker
import seaborn as sns

from xai_green_tech_adoption.utils.utils import *

def plt_performance(df_performances, perf_metric_test, perf_metric_train, list_runs, run_red_model, x_max, x_min=0, s=65, feat_count_red_model=None,
                    include_train_score=False,indicate_red_model=False):
    '''
    Function generates a matplotlib figure containing a plot of the r2-score depending on the number of features
    included in a model.
    @param df_performances: Dataframe giving details of hpo and rfe: the hyperparameters of the models and the
                            resulting performances (r2-score on training and test set and resulting ranking)
    @param list_runs:
    @param col_run:
    @param run_red_model:
    @param col_feature_count: (str) column name of number of features considered by model
    @parammean_r2_cv_test: (str) column name of r2-score of test data
    @param r2_train: (str) column name of r2-score of training data
    @param feat_count_red_model: number of input features considered by the reduced model
    @param include_train_score: (boolean) indicating whether diagram contains r2-scores of training data, default: False
    @param include_all_runs: (boolean) indicating whether diagram contains r2-scores of all or only best models,
                                default: True (all models included)
    @param indicate_red_model: (boolean) indicating whether r2 score(s) of reduced model are indicated by marker
    @return: plot instance
    '''
    df_performances = df_performances[df_performances[ranking_mean_r2_desc] == 1]

    fig = plt.figure(figsize=(7.5,5))

    ax = fig.add_subplot(111)
    label_alternative_runs_test = True
    label_alternative_runs_train = True
    for run in list_runs:
        df_perf_run = df_performances[df_performances[col_run_id]==run]
        if run==run_red_model:
            ax.scatter(x=df_perf_run[col_feature_count], y=df_perf_run[perf_metric_test], c='tab:green', marker='.', label='test fold (selected split)', s=s)
        else:
            if label_alternative_runs_test:
                ax.scatter(x=df_perf_run[col_feature_count], y=df_perf_run[perf_metric_test], c='tab:green', marker='.', label='test fold (other splits)',
                           alpha=0.1, s=s)
                label_alternative_runs_test = False
            else:
                ax.scatter(x=df_perf_run[col_feature_count], y=df_perf_run[perf_metric_test], c='tab:green', marker='.', alpha=0.1, s=s)

        if include_train_score:
            if run == run_red_model:
                ax.scatter(x=df_perf_run[col_feature_count], y=df_perf_run[perf_metric_train], c='royalblue', marker='.', label='training folds (selected split)',
                       s=s)
            else:
                if label_alternative_runs_train:
                    ax.scatter(x=df_perf_run[col_feature_count], y=df_perf_run[perf_metric_train], c='royalblue', marker='.', label='training folds (other splits)',
                           s=s,alpha=0.1)
                    label_alternative_runs_train = False
                else:
                    ax.scatter(x=df_perf_run[col_feature_count], y=df_perf_run[perf_metric_train], c='royalblue', marker='.',s=s,alpha=0.1)

    if indicate_red_model:
        df_red_model = df_performances.loc[
            (df_performances[ranking_mean_r2_desc] == 1) & (df_performances[col_feature_count] == feat_count_red_model) & (df_performances[col_run_id]==run_red_model)]
        if include_train_score:
            ax.scatter(x=df_red_model[col_feature_count], y=df_red_model[perf_metric_train], c='darkred', marker='x', s=100)
        ax.scatter(x=df_red_model[col_feature_count], y=df_red_model[perf_metric_test], c='darkred', marker='x',
                   label='scores of reduced model', s=100)
    plt.axvline(x=feat_count_red_model, color='darkred', linestyle='--',linewidth=2.5)
    ax.set_xlabel('number of features', fontsize=20,labelpad=10)
    fontsize_y = 20
    if perf_metric_test==mean_r2_cv_test:
        ax.set_ylabel(r'mean $R^2$ score', fontsize=fontsize_y,labelpad=10)
    elif perf_metric_test==mean_mae_cv_test:
        ax.set_ylabel('mean MAE', fontsize=fontsize_y)
    else:
        ax.set_ylabel('mean mape', fontsize=fontsize_y)

    ax.tick_params(labelsize=18)
    plt.yticks([0.6,0.7,0.8,0.9,1])
    ax.set_ylim((0.5,1.02))
    if x_max!=None:
        ax.set_xlim((x_min,x_max))
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2,3,0,1,4]
    if perf_metric_test==mean_r2_cv_test:
        plt.legend(handles=[handles[idx] for idx in order],
                   labels=[labels[idx] for idx in order], 
                   loc='lower right', fontsize=14,markerscale=2)
    else:
        plt.legend(handles=[handles[idx] for idx in order],
                   labels=[labels[idx] for idx in order],
                   loc='upper right', fontsize=14,markerscale=2)
    plt.tight_layout()

    return fig, ax


def ax_performance(ax, df_performances, perf_metric_test, perf_metric_train, 
                   list_runs, run_red_model, x_max, x_min=0, s=65, 
                   feat_count_red_model=None,
                   include_train_score=False,
                   indicate_red_model=False, label_yaxis=True,
                   show_legend=True):
    '''
    Function generates a matplotlib figure containing a plot of the r2-score depending on the number of features
    included in a model. This variant just plots_it on an existing axis.
    @param df_performances: Dataframe giving details of hpo and rfe: the hyperparameters of the models and the
                            resulting performances (r2-score on training and test set and resulting ranking)
    @param list_runs:
    @param col_run:
    @param run_red_model:
    @param col_feature_count: (str) column name of number of features considered by model
    @parammean_r2_cv_test: (str) column name of r2-score of test data
    @param r2_train: (str) column name of r2-score of training data
    @param feat_count_red_model: number of input features considered by the reduced model
    @param include_train_score: (boolean) indicating whether diagram contains r2-scores of training data, default: False
    @param include_all_runs: (boolean) indicating whether diagram contains r2-scores of all or only best models,
                                default: True (all models included)
    @param indicate_red_model: (boolean) indicating whether r2 score(s) of reduced model are indicated by marker
    @return: plot instance
    '''
    
    df_performances = df_performances[df_performances[ranking_mean_r2_desc] == 1]

    label_alternative_runs_test = True
    label_alternative_runs_train = True
    for run in list_runs:
        df_perf_run = df_performances[df_performances[col_run_id]==run]
        if run == run_red_model:
            ax.scatter(x=df_perf_run[col_feature_count], 
                       y=df_perf_run[perf_metric_test], c='tab:green', 
                       marker='.', label='test fold (selected split)', s=s)
        else:
            if label_alternative_runs_test:
                ax.scatter(x=df_perf_run[col_feature_count], 
                           y=df_perf_run[perf_metric_test], c='tab:green', 
                           marker='.', label='test fold (other splits)',
                           alpha=0.1, s=s)
                label_alternative_runs_test = False
            else:
                ax.scatter(x=df_perf_run[col_feature_count], 
                           y=df_perf_run[perf_metric_test], 
                           c='tab:green', marker='.', alpha=0.1, s=s)

        if include_train_score:
            if run == run_red_model:
                ax.scatter(x=df_perf_run[col_feature_count], 
                           y=df_perf_run[perf_metric_train], 
                           c='royalblue', marker='.', 
                           label='training folds (selected split)',
                       s=s)
            else:
                if label_alternative_runs_train:
                    ax.scatter(x=df_perf_run[col_feature_count], y=df_perf_run[perf_metric_train], c='royalblue', marker='.', label='training folds (other splits)',
                           s=s,alpha=0.1)
                    label_alternative_runs_train = False
                else:
                    ax.scatter(x=df_perf_run[col_feature_count], y=df_perf_run[perf_metric_train], c='royalblue', marker='.',s=s,alpha=0.1)

    if indicate_red_model:
        df_red_model = df_performances.loc[
            (df_performances[ranking_mean_r2_desc] == 1) & (df_performances[col_feature_count] == feat_count_red_model) & (df_performances[col_run_id]==run_red_model)]
        if include_train_score:
            ax.scatter(x=df_red_model[col_feature_count], y=df_red_model[perf_metric_train], c='darkred', marker='x', s=100)
        ax.scatter(x=df_red_model[col_feature_count], y=df_red_model[perf_metric_test], 
                   c='darkred', marker='x',
                   label='scores of reduced model', s=100)
        
    ax.axvline(x=feat_count_red_model, color='darkred', 
               linestyle='--',linewidth=2.5)
    ax.set_xlabel('number of features', fontsize=20,labelpad=10)

    if label_yaxis:
        fontsize_y = 20
        if perf_metric_test==mean_r2_cv_test:
            ax.set_ylabel(r'mean $R^2$ score', fontsize=fontsize_y,labelpad=10)
        elif perf_metric_test==mean_mae_cv_test:
            ax.set_ylabel('mean MAE', fontsize=fontsize_y)
        else:
            ax.set_ylabel('mean mape', fontsize=fontsize_y)

    ax.tick_params(labelsize=18)
    ax.set_yticks([0.6,0.7,0.8,0.9,1])
    ax.set_ylim((0.5,1.02))
    
    if x_max!=None:
        ax.set_xlim((x_min,x_max))
    handles, labels = ax.get_legend_handles_labels()
    order = [2,3,0,1,4]

    if show_legend:
        if perf_metric_test==mean_r2_cv_test:
            ax.legend(handles=[handles[idx] for idx in order],
                    labels=[labels[idx] for idx in order], 
                    loc='lower right', fontsize=16,markerscale=2)
        else:
            ax.legend(handles=[handles[idx] for idx in order],
                    labels=[labels[idx] for idx in order],
                    loc='upper right', fontsize=16,markerscale=2)

    return


def hbar_shap_feature_imp(df_run_eval_input, features):
    '''
    Create horizontal bar plot of mean SHAP feature importances over all ten training-test-splitting. The color of the bars
    indicates the number of reduced models a features occurs in.
    @param df_run_eval_input: dataframe of SHAP feature importances returned by get_mean_shap()
    @param features: list of input features
    @return: figure and axis giving the plot
    '''
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    
    # prepare_performance_dataframe
    df_run_eval = df_run_eval_input.copy()

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
                        'share 4-person households': '4-person households',
                        'share 5-person households': '5-person households',
                        'CDU/CSU': 'votes CDU/CSU (conservative party)'
                        }
    
    df_run_eval.rename(rename_tick_dict, axis=0, inplace=True)
    features = [rename_tick_dict[feat] if feat in rename_tick_dict else feat for feat in features]
    norm = plt.Normalize(df_run_eval.loc[features, col_occurences_feat].min(),
                         df_run_eval.loc[features, col_occurences_feat].max())
    cmap = plt.get_cmap('coolwarm')

    features = features[::-1]
    ax.bar(x=np.arange(len(features)),
            height=df_run_eval.loc[features, col_mean_shap],
            yerr=df_run_eval.loc[features, col_std_shap],
            color=cmap(norm(df_run_eval.loc[features, col_occurences_feat])))
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(features, size=16)
    ax.set_ylabel(r'mean(SHAP feature importance)', size=18)
    ax.margins(x=0.02)
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(df_run_eval.loc[features, col_occurences_feat].min(),df_run_eval.loc[features, col_occurences_feat].max()))
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, shrink=.45, anchor=(0.9, 10), 
                        orientation="horizontal")
    cbar.set_label('number of runs', rotation=360, 
                   labelpad=10, size=18)
    cbar.ax.tick_params(labelsize=16)

    plt.xticks(rotation=70,ha='right')
    plt.yticks(size=16)
    plt.tight_layout()

    return fig, ax


def bar_shap_feature_imp(df_run_eval_input: pd.DataFrame, features, ax: plt.axis=None, 
                         nr_features_shown: int = None, feature_occurence_lims: tuple=None,
                         cmap: str = 'coolwarm', labelsize: float =16., xlabel: str = None,
                         scalefactor_x=1.):
    '''
    Create bar plot of mean SHAP feature importances over all ten training-test-splitting. The color of the bars
    indicates the number of reduced models a features occurs in.
    @param df_run_eval_input: dataframe of SHAP feature importances returned by get_mean_shap()
    @param features: list of input features
    @return: figure and axis giving the plot
    '''

    if ax is None:
        fig = plt.figure(figsize=(10, 17))
        ax = fig.add_subplot(111)

    df_run_eval = df_run_eval_input.copy()

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
                        'CDU/CSU': 'votes CDU/CSU (conservative party)',
                        'other parties': 'votes other parties',
                        'FDP': 'votes FDP',
                        'The Left': 'votes The Left',
                        'AfD': 'votes AfD',
                        'The Greens': 'votes The Greens',
                        'SPD': 'votes SPD'}
    
    df_run_eval.rename(rename_tick_dict, axis=0, inplace=True)
    features = [rename_tick_dict[feat] 
                if feat in rename_tick_dict else feat for feat in features]
    
    if feature_occurence_lims is None:
        norm = plt.Normalize(df_run_eval.loc[features, 
                                            col_occurences_feat].min(),
                            df_run_eval.loc[features, 
                                            col_occurences_feat].max())
    else:
        norm = plt.Normalize(min(feature_occurence_lims),
                             max(feature_occurence_lims))
    max_runs = df_run_eval[col_occurences_feat].max()
    min_runs = df_run_eval[col_occurences_feat].min()
    cmap = plt.get_cmap(cmap, int(max_runs - min_runs) +1 )

    if nr_features_shown is None:
        widths = df_run_eval.loc[features, col_mean_shap]
        xerr = df_run_eval.loc[features, col_std_shap]
    else:
        widths = df_run_eval.loc[features, col_mean_shap].iloc[-nr_features_shown:]
        xerr = df_run_eval.loc[features, col_std_shap].iloc[-nr_features_shown:]
        features = features[-nr_features_shown:]

    ax.barh(y=np.arange(len(features)),
            width=widths*scalefactor_x,
            xerr=xerr*scalefactor_x,
            color=cmap(norm(df_run_eval.loc[features, col_occurences_feat])))
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features, size=12)
    if xlabel is None:
        ax.set_xlabel('mean $| \\textrm{SHAP values} |$', size=labelsize)
    else:
        ax.set_xlabel(xlabel, size=labelsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.locator_params(axis='x', nbins=4)
    ax.margins(x=0.02)

    bounds = np.arange(df_run_eval.loc[features, col_occurences_feat].min(),
                       df_run_eval.loc[features, col_occurences_feat].max() + 2, 1)

    sm = ScalarMappable(cmap=cmap, norm=mcolors.BoundaryNorm(bounds, cmap.N))
    sm.set_array([])

    if ax is None:
        cbar = plt.colorbar(sm, shrink=.45, anchor=(-1.5, 0.1))
        cbar.set_label('number of runs', rotation=270, labelpad=25, size=labelsize)
        cbar.ax.tick_params(labelsize=labelsize)

    ax.tick_params(labelsize=labelsize)
    
    if ax is None:
        plt.tight_layout()
        return fig, ax
    
    else:
        return sm
    

def dependence_plot(X, shap_values, feature, 
                    interaction_feature=None, lasso_mean=None, 
                    lasso_std=None, lasso_coef=None,
                    y_lim=None, x_lim=None, y_label=None, 
                    x_label=None, cb_label=None, title=None, 
                    scatter_color='royalblue',fig_size=(8, 5.5),
                    ax=None, label_size: float = 30., label_pad: float = 10.,
                    scalefactor_y: float = 1.):
    '''
    Create SHAP dependence plot of feature. Optional colouring according to interaction_feature.
    @param X: Input dataset to get indices of features
    @param shap_values: matrix of SHAP values
    @param feature: feature under investigation
    @param interaction_feature: (optional) interacting features used for colouring
    @param lasso_mean: mean of feature values used for standardisation of data to plot SHAP values of lasso
    @param lasso_std: standard deviation of feature used for standardisation of data to plot SHAP values of lasso
    @param lasso_coef: coefficient of the feature of the lasso model
    @param y_lim: (optional) tuple indicating limits of y-axis
    @param x_lim: (optional) tuple indicating limits of x-axis
    @param y_label: (optional) label of y-axis
    @param x_label: (optional) label of x-axis
    @param cb_label: (optional) label of colorbar
    @param title: title of figure
    @param scatter_color: color of scatter plot if no interaction feature is included
    @param fig_size: size of figure
    @param ax: (optional) axis to plot on. If 'None', a new figure will be produced.
    @return: figure and axis of plot
    '''
    # plot shap values vs feature values, coloring determined by values of interaction_feature
    feature_idx = list(X.columns).index(feature)
    if ax is None:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)

    label_pad = 10
    if (lasso_std != None) & (lasso_mean != None) & (lasso_coef != None):
        plt.plot(X[feature], lasso_coef * (X[feature] - lasso_mean) / lasso_std, color='darkgray', linestyle='-',
                 alpha=0.7,linewidth=2)
    if interaction_feature != None:
        scatter = ax.scatter(x=X[feature], y=scalefactor_y * shap_values[:, feature_idx], 
                             c=X[interaction_feature], cmap='coolwarm',
                             s=2, alpha=1)
    else:
        scatter = ax.scatter(x=X[feature], 
                             y=scalefactor_y * shap_values[:, feature_idx], 
                             c=scatter_color, s=2, alpha=1)
    if x_lim != None:
        ax.set_xlim(x_lim)
    if y_lim != None:
        ax.set(ylim=y_lim)
    if x_label != None:
        ax.set_xlabel(x_label, fontsize=label_size, labelpad=label_pad)
    else:
        ax.set_xlabel(feature, fontsize=label_size, labelpad=label_pad)
    if y_label != None:
        ax.set_ylabel(y_label, fontsize=label_size, labelpad=label_pad)
    else:
        ax.set_ylabel(f'SHAP values of\n{feature}', 
                      fontsize=label_size, labelpad=label_pad)
        
    ax.tick_params('both', labelsize=label_size-2)

    if feature=='regional potential of population':
        ax.locator_params(axis='x', nbins=4)
    else:
        ax.locator_params(axis='both', nbins=6)

    if title != None:
        ax.title(title, fontsize=label_size+2)

    if interaction_feature != None:
        cbar = plt.colorbar(scatter)
        if cb_label!=None:
            cbar.set_label(label=cb_label,size=label_size, labelpad=label_pad)
        else:
            cbar.set_label(label=interaction_feature, size=label_size-2, labelpad=label_pad)
        cbar.ax.tick_params(labelsize=label_size-2)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
    
    if ax is None:    
        return fig, ax
    else:
        return
    

def heatmap_interactions(X, interaction_values, feature_name_dict: dict  = None, 
                         zero_main_effect: bool = True, 
                         ax: plt.axes = None, fontsize: float = 15., 
                         plot_cbar: bool = True, cmap: str='plasma', vlims: tuple = None,
                         remove_diagonal: bool = False, cbar_label: str = None,
                         scalefactor_cbar: float = 1.): 
    '''
    Create heat map of SHAP interaction values.
    @param X: Input data
    @param interaction_values: Interaction values
    @param feature_name_dict: dictionary giving feature names to modify
    @param zero_main_effect: Boolean indicating whether the main effects 
        should be set to zero. Default is true.
    @return: figure and axis giving heat map
    '''
    features = list(X.columns)
    if feature_name_dict != None:
        features_renamed = [feature_name_dict[feature] if feature in feature_name_dict 
                            else feature for feature in
                            features]
    else:
        features_renamed = features

    mean_interactions = abs(interaction_values).mean(axis=0)

    if zero_main_effect:
        mean_interactions[np.arange(len(features_renamed)), np.arange(len(features))] = 0

    if remove_diagonal:
        mean_interactions[np.arange(len(features_renamed)), np.arange(len(features))] = np.nan

    kwargs = dict(cmap=cmap, cbar=plot_cbar, 
                  linewidths=0.5, linecolor='black')
    if vlims is not None:
        kwargs['vmin'] = min(vlims)
        kwargs['vmax'] = max(vlims)

    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = sns.heatmap(np.flip(mean_interactions) * scalefactor_cbar, **kwargs)

    else:
        sns.heatmap(np.flip(mean_interactions) * scalefactor_cbar, ax=ax, **kwargs)
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)

    ax.xaxis.tick_top()

    ax.set_xticklabels(features_renamed[::-1], rotation=90, size=fontsize)
    ax.set_yticklabels(features_renamed[::-1], size=fontsize)

    if plot_cbar:
        cb_ax = ax.figure.axes[-1]
        cb_ax.tick_params(labelsize=16)
        if cbar_label is None:
            cb_ax.set_ylabel('mean $|\\text{SHAP interaction value}|$', fontsize=24)
        else:
            cb_ax.set_ylabel(cbar_label, fontsize=24)

    ax.tick_params('y', rotation=0)

    plt.tight_layout()

    if ax is None:
        return fig, ax
    else:
        return
    

def dependence_plot_main_effect(X, shap_values, feature, plot_main_effect=False, 
                                y_lim=None, x_lim=None, y_label=None,
                                x_label=None, fig_size=(5,4), font_size = 12, 
                                tick_size = 12, ax=None):
    '''
    Scatter plot of SHAP values or main effects of SHAP values.
    @param X: input data, used to collect indices of features
    @param shap_values: if plot_main_effect is True: matrix of SHAP interaction values, else: matrix of SHAP values
    @param feature: feature to be analysed
    @param plot_main_effect: boolean indicating whether the main effect (or the simple SHAP values) should be plotted
    @return: figure and axis element the figure is added to
    '''
    feature_idx = list(X.columns).index(feature)
    if plot_main_effect:
        shap_to_plot = shap_values[:,feature_idx,feature_idx]
    else:
        shap_to_plot = shap_values[:,feature_idx]

    if ax is None:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
    
    ax.scatter(x=X[feature], y=shap_to_plot, c='silver', s=1, alpha=0.5)

    if x_lim != None:
        ax.set_xlim(x_lim)
    if y_lim != None:
        ax.set(ylim=y_lim)
    if x_label != None:
        ax.set_xlabel(x_label, fontsize=font_size)
    else:
        ax.set_xlabel(feature, fontsize=font_size)
    
    if y_label != None:
        ax.set_ylabel(y_label, fontsize=font_size)
    #ax.set_title(y_label, fontsize=font_size+1)

    ax.tick_params('both', size=tick_size)

    ax.locator_params(axis='both', nbins=6)

    if ax is None:
        plt.tight_layout()

        return fig, ax
    else:
        return

def dependence_plot_interactions(X, interaction_vals, feature, interaction_feature, 
                                 ax, x_label=None, y_label=None,
                                 cb_label=None, x_lim=None, y_lim=None, 
                                 y_ticks=False, title=None, font_size=12, tick_size = 12,
                                 cmap: str = 'inferno'):
    '''
    Scatter plot of SHAP interaction values.
    @param X: input data to derive the indices of the feature from
    @param interaction_vals: 3D matrix of interaction values
    @param feature: main feature displayed on x-axis
    @param interaction_feature: interacting feature used for colouring
    @param ax: axis item to add the plot to
    @return: scatter plot and axis element the plot is added to
    '''

    feature_idx = list(X.columns).index(feature)
    interaction_idx = list(X.columns).index(interaction_feature)
    scatter = ax.scatter(x=X[feature], y=interaction_vals[:, feature_idx, 
                                                          interaction_idx], 
                         c=X[interaction_feature],
                         cmap=cmap, s=1, alpha=0.5)
    
    if x_label != None:
        ax.set_xlabel(x_label, fontsize=font_size)
    if y_label != None:
        ax.set_ylabel(y_label, fontsize=font_size)
    if title != None:
        ax.set_title(title, fontsize=font_size)
    else:
        ax.set_title(f'interaction with\n {interaction_feature}', fontsize = font_size)

    ax.tick_params(labelsize=tick_size)

    if x_lim != None:
        plt.xlim(x_lim)

    if y_lim != None:
        plt.ylim(y_lim)

    if cb_label != None:
        cbar = plt.colorbar(scatter)
        cbar.set_label(label=cb_label, size=font_size)
        cbar.ax.tick_params(labelsize=font_size)

    return scatter, ax

def bar_mean_shap(X, shap_values, bar_color='royalblue',rename_features_dict=None, x_lim=None, show_std=True,title=None):
    '''
    Create bar plot of global SHAP features importances (mean(|SHAP value|) and the standard deviation of the
    absolute values of the SHAP values of a (GBT) model.
    @param X: dataframe of input data (used to obtain feature names)
    @param shap_values: SHAP values
    @param rename_features_dict: (optional, default = None) dictionary indicating changes of feature name to be
                                    displayed in the plot
    @param x_lim: (optional) tuple indicating limits of x-axis
    @return: bar plot given by figure and axis element
    '''
    features = list(X.columns)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx_shap = np.argsort(mean_abs_shap)
    std_abs_shap = np.abs(shap_values).std(axis=0)
    mean_abs_shap = mean_abs_shap[sorted_idx_shap]
    std_abs_shap = std_abs_shap[sorted_idx_shap]
    features = np.array(features)[sorted_idx_shap]

    if rename_features_dict != None:
        feature_labels = [rename_features_dict[feature] if feature in rename_features_dict else feature for feature in
                          features]
    else:
        feature_labels = features

    fig = plt.figure(figsize=(7,7.5))
    ax = fig.add_subplot(111)

    if show_std:
        ax.barh(y=np.arange(len(features)), color=bar_color, width=mean_abs_shap, xerr=std_abs_shap)
    else:
        ax.barh(y=np.arange(len(features)), color=bar_color, width=mean_abs_shap)
    ax.set_yticks(np.arange(len(features)), labels=feature_labels)
    ax.set_xlabel(r'mean(|SHAP value|)',fontsize=16)
    ax.tick_params(axis='x',labelsize=16)
    ax.tick_params(axis='y', labelsize=14)

    if x_lim != None:
        plt.xlim(x_lim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if title != None:
        plt.title(title, fontsize=18)
    plt.tight_layout()

    return fig, ax

def bar_shap_reduced(X, shap_values, bar_color='royalblue',
                     n_print=None,
                     x_label=None,
                     x_lim=None, 
                     show_std:bool=True, 
                     draw_values_in_bar: bool = True, ax=None):
    '''
    Plotting SHAP feature importances of most important features and 
    sum of remaining SHAP feature importance.

    @param X: Data the model is trained on (dataframe)
    @param shap_values: shap_values of model
    @param n_print: Number of features to be displayed
    @return: figure and axis
    '''
    features = list(X.columns)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx_shap = np.argsort(mean_abs_shap)
    std_abs_shap = np.abs(shap_values).std(axis=0)
    mean_abs_shap = mean_abs_shap[sorted_idx_shap]
    std_abs_shap = std_abs_shap[sorted_idx_shap]

    if n_print is not None:
        mean_abs_shap = mean_abs_shap[:n_print]
        std_abs_shap = std_abs_shap[:n_print]

    features = np.array(features)[sorted_idx_shap]
    features = features[-n_print:]

    if ax is None:
        fig = plt.figure(figsize=(7,4))
        ax = fig.add_subplot(111)

    if show_std:
        ax.barh(y=np.arange(len(features)), 
                color=bar_color, width=mean_abs_shap, 
                xerr=std_abs_shap)
    else:
        ax.barh(y=np.arange(len(features)), 
                color=bar_color, 
                width=mean_abs_shap,
                height = 0.7)

    ax.set_xticks([0,0.01])
    ax.set_yticks(np.arange(len(features)), 
                  labels=features)
    
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=20)

    ax.tick_params(axis='y',labelsize=20)
    ax.tick_params(axis='x',labelsize=27)
    
     
    if x_lim != None:
        ax.set_xlim(x_lim)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if ax is None:
    
        return fig, ax
    else:
        return


def plot_performance_lasso(df_lasso_mean, perf_metric_train, perf_metric_test, 
                           x_max=None, fig_size=(9, 5),
                           y_min=0, y_max=0.9, threshold_inv_alpha=None, 
                           pos_leg_in_figure=True,
                           ax=None):
    '''
    Plot mean performance metric (R2 score, MAE, MAPE or MSE) over the inverse of alpha.
    @param df_lasso: dataframe with mean performances over all train-test-splits
    @param perf_metric_train: column name of the column containing the values of the performance metric on the training set
    @param perf_metric_test: column name of the column containing the values of the performance metric on the test set
    @param list_runs: list of all simulation runs
    @param run_red_model: number of simulation used for reduced model (in this case: 4th training-test-split)
    @param threshold_inv_alpha: threshold of inverse alpha to determine reduced model (if None, threshold will not be indicated)
    @param fig_size: size of the figure
    @param ax: (optional) axis to plot on. If 'None', a new figure will be produced.
    @return: figure instance with plot of coefficients
    '''

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)

    ax.scatter(x=1 / df_lasso_mean[col_alpha], y=df_lasso_mean[perf_metric_train], 
               c='tab:blue', label='training set',
                s=1)
    ax.scatter(x=1 / df_lasso_mean[col_alpha], y=df_lasso_mean[perf_metric_test], 
               c='tab:green', label='test set', s=1)
    ax.set_xlim((0, x_max))
    ax.set_ylim(y_min, y_max)

    if threshold_inv_alpha != None:
        ax.axvline(x=threshold_inv_alpha, color='darkred', linestyle='--')

    if ax is None:
        if perf_metric_test == col_r2_test:
            ax.set_ylabel(r'mean $R^2$ score', fontsize=16, loc='center')
        elif perf_metric_test == col_mape_test:
            ax.set_ylabel('mean mape', fontsize=16, loc='center')
        elif perf_metric_test == col_mae_test:
            ax.set_ylabel('mean mae ', fontsize=16, loc='center')
        elif perf_metric_test == col_mse_test:
            ax.set_ylabel('mean mse', fontsize=16, loc='center')
        else:
            raise ValueError('Unknown Performance Metric.')
        
        ax.tick_params(labelsize=13)
        ax.set_xlabel(r'$\alpha^{-1}$', fontsize=16)

        if not pos_leg_in_figure:
            fig_box = ax.get_position()
            ax.set_position([fig_box.x0, fig_box.y0, fig_box.width * 0.7, fig_box.height])
            ax.set_anchor('SW')
            legend = ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), 
                            markerscale=5, fontsize=12)
        elif perf_metric_test == col_r2_test:
            legend = ax.legend(loc='lower right', markerscale=5, fontsize=13)
        else:
            legend = ax.legend(loc='upper right', markerscale=5, fontsize=13)
    
    
        # Aesthetics
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        return fig, ax, legend
    
    else:  
        return
    

def plot_count_non_zero_coef(df_lasso_input, col_alpha, 
                             x_max=None, y_min=None, y_max=None, 
                             threshold_inv_alpha=None,
                             fig_size=None, ax=None, color='C1'):
    '''
    Plot mean number of non-zero features of all ten reduced lasso models over the inverse of alpha.
    @param df_lasso_input: dataframe giving results of simulations of lasso models, one row corresponds to one simulated
                            lasso model
    @param col_alpha: column name of column giving values of alpha of lasso model
    @param x_max: (optional) maximum value of x-axis displayed
    @param y_min: (optional) minimum value of y-axis displayed
    @param y_max: (optional) maximum value of y-axis displayed
    @param threshold_inv_alpha: choice of the inverse of alpha of the reduced lasso model
    @param fig_size: size of the figure
    @param ax: (optional) axis to plot on. If 'None', a new figure will be produced.
    @return: figure containing the plot
    '''

    df_lasso = df_lasso_input.copy()

    if ax is None:
        if fig_size != None:
            fig = plt.figure(figsize=fig_size)
        else:
            fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.step(x=1 / df_lasso[col_alpha], 
            y=df_lasso[col_count_non_zero], color=color)

    if threshold_inv_alpha != None:
        ax.axvline(x=threshold_inv_alpha, color='darkred', linestyle='--')

    if ax is None:
        plt.ylabel('number of features with\nnon-zero coefficients', fontsize=14)
        plt.yticks(fontsize=12)
        if (y_min != None) & (y_max != None):
            plt.ylim((0, 30))
        plt.xlabel(r'$\alpha^{-1}$', fontsize=14)
        plt.xticks(fontsize=12)
        if x_max != None:
            plt.xlim((0, x_max))


        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        return fig
    
    else:
        return


def reduce_df_size(df, max_datapoints):
    '''
    Reduced number of datapoints contained in the dataframe df to max_datapoints. (Used to plot coefficients over
    inverse alpha).
    @param df: dataframe containing simulations
    @param max_datapoints: maximum number of simulations to be contained in the dataframe after the reduction
    @return: dataframe with reduced number of simuations
    '''
    count_datapoint = df.shape[0]
    ratio = np.ceil(count_datapoint / max_datapoints)
    # every ratio-th element will be kept
    idx_to_keep = df.index[df.index % ratio == 0]
    return df.iloc[idx_to_keep, :]


def plot_mean_coefficients(df_lasso_mean: pd.DataFrame, feature_list: list, 
                           x_max: float = None, threshold_inv_alpha: float = None,
                           ax: plt.axes=None, feature_rename_dict: dict = None, 
                           verbose: bool = False):
    '''
    Plotting mean coefficients of lasso models over the inverse of alpha.
    @param df_lasso_mean: Dataframe giving mean results and mean coefficients of all simulation runs. One row
            corresponds to a simulation run for one value of alpha. The dataframe contains the mean values over all ten
            training-test-splittings for all values of alpha.
    @param feature_list: list of all input features considered which also make up column names of columns giving
            coefficients of respective features.
    @param x_max: maximum x value displayed
    @param threshold_inv_alpha: threshold of inverse alpha. Basis for choosing reduced lasso model.
    @return: figure and legend giving the plot
    '''
    features_occurring_list = []
    min_inverse_alpha = []
    feature_label_colors_dict = {}
    # get list of features that have non-zero coef for 
    # any value of inverse alpha displayed in plot
    for feature in feature_list:
        if (df_lasso_mean[feature] != 0).any():
            min_inverse_alpha_feat = 1. / df_lasso_mean.loc[(df_lasso_mean[feature] != 0) & (
                    df_lasso_mean[col_alpha] == df_lasso_mean.loc[
                (df_lasso_mean[feature] != 0), col_alpha].max()), col_alpha].values[0]
            if min_inverse_alpha_feat <= x_max:
                features_occurring_list += [feature]
                min_inverse_alpha += [min_inverse_alpha_feat]
                if min_inverse_alpha_feat <= threshold_inv_alpha:
                    feature_label_colors_dict[feature] = 'black'
                else:
                    feature_label_colors_dict[feature] = 'darkgray'

        if verbose:
            print(f"{feature}, min_inverse_alpha: {min_inverse_alpha_feat}")

    # sort features according to their order of appearance
    idx_sorted = np.argsort(min_inverse_alpha)
    min_inverse_alpha = np.array(min_inverse_alpha)[idx_sorted]
    features_occurring_list = np.array(features_occurring_list)[idx_sorted]

    # rename features
    feature_label_dict = {'Certificates of Secondary Education (males)': 'certificates of secondary\neducation (males)',
                          'per capita permissions for (semi-) detached houses ': 'per capita permissions for (semi-)\ndetached houses',
                          'completed (semi-) detached houses (per capita)': 'completed (semi-) detached\nhouses (per capita)',
                          'distance public transport max 1km': 'distance public transport\nmax 1km',
                          'apprentices per young inhabitant': 'apprentices per young\ninhabitant',
                          'employees in knowledge-intensive industries': 'employees in knowledge-\nintensive industries',
                          'completed flats with renewable heat energy systems': 'completed flats with renew-\nable heat energy systems',
                          'CDU/CSU': 'votes CDU/CSU (conservative party)',
                          'other parties': 'votes other parties'
                          }
    
    feature_labels = [feature_label_dict[feature] 
                      if feature in feature_label_dict else feature for feature in
                      features_occurring_list]
    
    feature_label_colors = [feature_label_colors_dict[feature] 
                            for feature in features_occurring_list]

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1.5 * 7.3, 1.5 * 4))

    color_map = plt.get_cmap('nipy_spectral')

    # set colors for ordered features
    ax.set_prop_cycle('color', [color_map(idx / len(min_inverse_alpha)) 
                                for idx, _ in enumerate(min_inverse_alpha)])

    for idx, feature in enumerate(features_occurring_list):
        if feature_rename_dict is not None:
            label_str = feature_rename_dict[feature] \
                if feature in feature_rename_dict else feature
        else:
            label_str = feature_labels[idx]

        ax.scatter(x=1. / df_lasso_mean[col_alpha], y=df_lasso_mean[feature], 
                   label=label_str,
                   marker='.', s=1)

    if ax is None:
        plt.axvline(x=threshold_inv_alpha, color='darkred', linestyle='--')
        fig_box = ax.get_position()
        ax.set_position([fig_box.x0, fig_box.y0, fig_box.width * 0.8, fig_box.height])
        ax.set_anchor('SW')
        legend = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                           markerscale=10, fontsize=10,
                           labelcolor=feature_label_colors)
        ax.set_xlim((0, x_max))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel(r'mean $\beta_j$', fontsize=16, loc='center')
        ax.tick_params(labelsize=14)
        ax.set_xlabel(r'$\alpha^{-1}$', fontsize=16)

        return fig, legend
    
    else:
        return feature_label_colors, feature_label_colors_dict
    

def plot_pv_stock(df_stock, col_year, col_installations, col_stock):
    '''
    Plotting newly installed PV systems and accumulated Stock of PV systems over time.
    @param df_stock: Dataframe with data to plot
    @param col_year: Column name with year of record
    @param col_installations: Column with new PV installations per year
    @param col_stock: Column with data on accumulated PV stock
    @return: figure and axes objects
    '''
    f_size = 22
    xtick_size = 17
    ytick_size = 20
    l_pad = 5
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax2 = ax1.twinx()
    ax1.bar(x=df_stock[col_year], height=df_stock[col_installations], width = 0.8, color='royalblue')
    ax2.plot(df_stock[col_year], df_stock[col_stock], lw=3.5, color ='firebrick')
    ax2.vlines(2008.48, ymin=0,ymax=14.2, linestyles='--',color='grey',linewidth=2)
    ax2.vlines(2012, ymin=0,ymax=14.2, linestyles='--',color='grey',linewidth=2)
    ax2.vlines(2021.48, ymin=0,ymax=14.2, linestyles='--',color='grey',linewidth=2)
    ax1.set_ylim(0,2.75)
    ax2.set_ylim(0,14.2)
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()
    ax1.set_ylabel('Yearly installed capacity [$GW_p$]',  fontsize = f_size)
    ax2.set_ylabel('Accumulated capacity [$GW_p$]',  fontsize = f_size)
    ax1.yaxis.labelpad = l_pad
    ax2.yaxis.labelpad = l_pad
    ax1.tick_params(axis='x', labelsize=xtick_size)
    ax1.tick_params(axis='y', labelsize=ytick_size)
    ax2.tick_params(axis='y', labelsize=ytick_size)
    ax1.set_xticks(df_stock[col_year])
    ax1.set_xticklabels(df_stock[col_year], rotation=65, ha='right', rotation_mode='anchor')
    ax1.margins(x=0.01)
    ax1.spines[['top']].set_visible(True)
    ax2.spines[['top']].set_visible(False)
    return fig, ax1, ax2

def plot_shap_over_time(df_top_features, n_to_plot, dict_time_str):
    '''
    Plotting features with largest SHAP feature importances for multiple time periods..
    @param df_top_features: Data to plot
    @param n_to_plot: Number of feature per time period to plot
    @param dict_time_str: Dictionary with strings of limits of time periods
    @return: figure and axis 
    '''
    bar_indent = 15
    bar_pos_relative = np.arange(n_to_plot)*5
    bar_pos = np.array([])
    bar_width = 3.5
    # ticksize = 15
    labelsize = 17
    idx_timespans = np.arange(np.floor(n_to_plot/2),n_to_plot*4,n_to_plot, dtype=int)

    # plot three most important features for all time periods
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.minorticks_on()
    for idx, timespan in enumerate(dict_time_str):
        bar_pos = np.append(bar_pos, bar_pos_relative + idx * bar_indent)
        ax.bar(bar_pos_relative + idx * bar_indent, height=df_top_features.loc[df_top_features[timespan].notna(), timespan], label=timespan, color ='firebrick', width=bar_width)
        if idx != 3:
            ax.axvline(bar_pos[(idx+1)*n_to_plot-1]+ bar_indent/(2*n_to_plot), color='black',linewidth=1)
    ax.set_xticks(bar_pos)
    ax.set_xticklabels(['completed (semi-)\ndetached houses' if feat=='completed (semi-) detached houses (per capita)' else feat for feat in df_top_features.index.to_list()], rotation=65, ha='right', va = 'center', rotation_mode='anchor')
    pos_timespans = bar_pos[idx_timespans]+0.01
    ax.set_xticks(pos_timespans, minor=True, labels=[dict_time_str[timespan] for timespan in dict_time_str])
    ax.tick_params(axis='x', which='minor', pad=-215, labelsize=14) 
    ax.tick_params(axis='both', labelsize=14) 
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylim(0,0.059)
    ax.set_ylabel('SHAP feature importance', size=labelsize)
    ax.margins(x=0.02)
    plt.grid(False)
    return fig, ax