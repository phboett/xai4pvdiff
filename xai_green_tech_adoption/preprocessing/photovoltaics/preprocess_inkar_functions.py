#!/usr/bin/env python
# -*- coding: utf-8 -*

import pandas as pd
import numpy as np
import os, glob


def fix_time_periods(df, t_column):
    '''
    Change time data from different formats (XXXX bis XXXX, 'XXXX/XXXX/...','XXXX-XXXX') to single years
    Values of t_column are str-valued.
    @param df: Dataframe
    @param t_column: Name of column with time/ year of record.
    @return: Dataframe df with a transformed time column.
    '''


    periods = df[t_column].unique()
    # time spans of more than one year, i.e., more than four digits
    per_to_fix = [per for per in periods if len(per) > 4]

    # get list of years for longer periods of time (per_to_fix)
    per_fixed = []
    for p in per_to_fix:
        if " bis " in p:
            per_fixed.append(list(range(int(p[:4]), int(p[-4:]) + 1)))
        elif "/" in p:
            per_fixed.append(p.split('/'))
        elif "-" in p:
            per_fixed.append(list(range(int(p[:4]), int(p[-4:]) + 1)))
        else:
            raise ValueError(f'{t_column} not handled properly.')

    # elements of list: lists of multiples of the Zeitbezug; dimension corresponds to the length of the time period
    # len(per_fixed[idx]): length of time period
    per_to_fix_rep = [[elem] * len(per_fixed[idx]) for idx, elem in enumerate(per_to_fix)]
    # make one list of all time span elements
    per_to_fix_complete = [elem for l in per_to_fix_rep for elem in l]
    # merge newly generates times (=years) to one list
    per_fixed_complete = [elem for l in per_fixed for elem in l]

    # Create data frame with matching Zeitbezügen and corresponding years
    df_periods = pd.DataFrame({t_column: per_to_fix_complete, f'{t_column} (complete)': per_fixed_complete})
    # add generated corresponding years of data collection and use them to substitute Zeitbezüge of more than one year
    # note: merge returns a row for each duplicate value of the time periods in per_fixed_complete
    df = df.merge(df_periods, how='left', on=t_column)
    # substitute old 'Zeitbezüge'
    df.loc[df[f'{t_column} (complete)'].notna(), t_column] = df[f'{t_column} (complete)']
    df.drop(f'{t_column} (complete)', axis=1, inplace=True)
    df[t_column] = df[t_column].astype(int)

    return df


def transform_inkar_df(df, id_col, ind_col, ind_list, value_col, t_col, t_period):
    '''
    Transforms data frame of INKAR data set. Input data frame: There is one row for each indicator, each spacial
    entity (of one kind) and each point in time. Output data frame: Data for one year (t_period), one row for each spacial entity.
    Columns: Metadata to spacial entities and indicators/features.
    :param df: Dataframe to be transformed
    :param id_col: Name of column giving id of spacial entities
    :param ind_col: Name of column containing (names of) indicators
    :param value_col: Column containing the value of the indicator
    :param t_col: Column containing the period of time of the data
    :param t_period: Year/ period of time we are interested in
    :return: df_trans: Dataframe. Rows correspond to spacial entities containing all information and all indicators/features available for the entity.
    '''

    # only keep data from year of interest
    df_t = df[(df[ind_col].isin(ind_list)) & (df[t_col] == t_period)]
    ind_t = df_t[ind_col].unique()

    if len(ind_t) != len(ind_list):
        print(f'Warning: Not all relevant features are included in the dataframe. They may be missing for time period {t_period}.')

    # df_t: Dataframe with data for year of interest
    # df_transformed: Dataframe with spatial ids and data on spatial entities (should be unique for a spatial entity)
    df_transformed = df_t.drop([ind_col, t_col, value_col], axis=1)
    # Check whether additional characteristics of spatial entities are unique
    if np.invert(df_transformed.groupby(id_col).nunique() == 1).sum().sum() != 0:
        raise Exception('Ambiguous information on spatial entities.')

    # Due to previously dropped columns there occurred duplicates
    df_transformed.drop_duplicates(inplace=True)

    # Add all indicators/features to dataframe
    for feature in ind_t:
        # iterate over all indicators
        # get ids of the spacial entities and values of feature
        df_feature = df_t.loc[df_t[ind_col] == feature, [id_col, value_col]]
        # add feature values
        df_transformed = df_transformed.merge(df_feature, on=id_col, how='left')

        # column name should correspond to feature name
        df_transformed.rename({value_col: feature}, inplace=True, axis=1)

    return df_transformed


def add_ind(df, df_inkar, ind_dict, id_col, ind_col, value_col, t_col):
    '''
    Adds INKAR indicators contained in ind_dict to dataframe df. df_inkar gives the relevant values in a raw
    INKAR-style format. ind_dict is a dictionary with the indicators to be added as keys and the corresponding time
    periods of values.
    @param df: The additional ind. are added to this compact dataframe (which is already transformed)
    @param df_inkar: raw INKAR-style dataframe which contains the relevant values of the indicators to be added
    @param ind_dict: indicators to be added. keys: indicator names, values: the respective periods of time
    @param id_col: column name with ids of spatial entities (needs to be identical in df and df_inkar)
    @param ind_col: column giving indicators names in INKAR dataframe
    @param value_col: column giving values in INKAR dataframe
    @param t_col: column giving time period of value in INKAR dataframe
    @return: dataframe in compact format; df with additional columns corresponding to indicators given by ind_dict
    '''

    for ind in ind_dict:
        df_inkar_ind = df_inkar[df_inkar[ind_col] == ind]
        df_to_add = transform_inkar_df(df_inkar_ind, id_col=id_col, ind_col=ind_col, ind_list=[ind],
                                       value_col=value_col, t_col=t_col,
                                       t_period=ind_dict[ind])
        df_to_add = df_to_add[[id_col, ind]]
        df = df.merge(df_to_add, on=id_col, how='left')
    return df


def identify_missing_ind(df, indicators):
    '''
    Checks which indicators have NaNs.
    @param df: df (compact style) to check
    @param indicators: ind to check completeness for
    @return:  list of tuples, tuples give indicators and corresponding number of NaNs occurrences.
    '''
    ind_missing = []
    for ind in indicators:
        if (ind in df.columns) & (sum(df[ind].isna()) != 0):
            ind_missing.append((ind, sum(df[ind].isna())))
    return ind_missing


def detect_missings(df, inds, cat_id, cat_ind, dependencies_dict, comments_dict, entire_df, types):
    '''
    Detect indicators contained as columns in dataframe df that have NaN values.
    @param df: Dataframe to check for NaNs
    @param inds: indicators to check
    @param cat_id: name of column containing ids
    @param cat_ind: name of column containing indicators (in returned dataframe containing missings)
    @param dependencies_dict: dictionary of dependencies if indicators I identified (e.g., unemployed (Arbeitslose)
    and unemployed women ('Arbeitslose Frauen'). The dependent indicators are the keys, the indicators they are
    dependent on are the values.
    @param comments_dict: comments on the indicators and possible reasons for NaNs
    @param entire_df: string stating whether entire (c and ma) or partial (ma or c only) dataframe is analysed
    @param types: string stating what types of spatial entities are analysed (municipality associations or counties)
    @return: dataframe listing the indicators with NaNs, the respective numbers of NaNs and dependencies of and
    comments on the inds
    '''
    
    print(f'++++++++++ Checking {entire_df} dataframe of {types} +++++++++++\n')
    ind_missing_incl_count = identify_missing_ind(df, inds)
    ind_missing = [m[0] for m in ind_missing_incl_count]
    ind_missing_counts = [m[1] for m in ind_missing_incl_count]
    count_spatial_entities = df[cat_id].nunique()
    count_indicators = len(inds)
    count_spatial_entities_missing = sum(df.isna().sum(axis=1) != 0)
    print(
        f'{len(ind_missing) / count_indicators} ({len(ind_missing)} out of {count_indicators}) of the indicators have '
        f'missing values.')
    print(
        f'{count_spatial_entities_missing / count_spatial_entities} ({count_spatial_entities_missing} out of '
        f'{count_spatial_entities}) of the {types} have at least one missing indicator.\n')
    df_missings = pd.DataFrame({cat_ind: ind_missing, f'Abs. number of NaNs': ind_missing_counts,
                                f'Rel. number of NaNs': np.array(
                                    ind_missing_counts) / count_spatial_entities})
    df_missings['Identified dependencies'] = df_missings[cat_ind].apply(
        lambda dependent_ind: dependencies_dict[dependent_ind] if dependent_ind in dependencies_dict.keys() else np.nan)
    df_missings['Further comments'] = df_missings[cat_ind].apply(
        lambda ind: comments_dict[ind] if ind in comments_dict.keys() else np.nan)
    
    print(df_missings)

    return df_missings


def search_substitutes_from_past(df_t, df_inkar, inds_missing, cat_id, cat_ind, cat_time, time_period, cat_value):
    '''
    Detect which NaNs can be replaced by values from earlier periods of time and return dictionary with indicators
    with NaNs as keys and dataframes with substitution values and respective period of time as values. The most
    recent value available is used as substitute.
    @param df_t: dataframe which is checked for NaNs
    @param df_inkar: (relevant parts of the) original INKAR dataframe
    @param inds_missing: indicators which have NaNs and should be checked (there may be inds with NaNs that need not
    to be checked because I substitute the values differently)
    @param cat_id: name of column with ids of spstial entities
    @param cat_ind: name of column containing nmes of indicators in INKAR dataframe
    @param cat_time: name of column containg time epriods in INKAR dataframe
    @param time_period: time period of valued contained in dataframe df_t
    @param cat_value: name of column containing values of indicators in INKAR dataframe
    @return: dictionary with possible substitutes form past time periods; keys: indicators, values: dataframes with
    substitution values and respective period of time as values
    '''
    # detect whether values can be substituted from earlier years
    substitution_values = {}
    for ind in inds_missing:
        # get ids of entities with NaN values for ind
        ids_missing = list(df_t.loc[df_t[ind].isna(), cat_id])
        # time periods ind is available for in general
        t_periods_ind = np.array(df_inkar.loc[df_inkar[cat_ind] == ind, cat_time].unique())
        t_earlier_periods = np.flip(np.sort(t_periods_ind[t_periods_ind < time_period]))
        # create empty dataframe which will be filled with possible substitutes
        df_values_missing = pd.DataFrame()
        for id in ids_missing:
            for t in t_earlier_periods:
                # iterate backwards over earlier time periods and get most recent available value
                df_substitute = df_inkar[
                    (df_inkar[cat_ind] == ind) & (df_inkar[cat_id] == id) & (df_inkar[cat_time] == t)]
                if df_substitute.shape[0] > 1:
                    # ensure that the substitute value is uniquely defined
                    raise ValueError(
                        f'Ambiguous information: Multiple values for indicator {ind} in time period {t} for spatial '
                        f'entity with id {id}.')
                elif df_substitute.shape[0] == 1:
                    if ~np.isnan(df_substitute.loc[df_substitute[cat_id] == id, cat_value].values[0]):
                        df_values_missing = pd.concat([df_values_missing,
                                                       pd.DataFrame({cat_id: id, cat_time: t, cat_value: df_substitute[
                                                           cat_value].loc[df_substitute[cat_id] == id]})],
                                                      ignore_index=True)
                        break
        substitution_values.update({ind: df_values_missing})

    return substitution_values


def insert_and_save_substitutes_from_past(df_preprocessed, df_inkar, inds_missing, cat_id, cat_ind, cat_time,
                                          time_period, cat_value, spatial_type):
    '''
    Searches NaNs if indicators given by inds_missing and substitute them with values from earlier periods of time if
    available. Saves the used substitutes in csv-files, one for each indicator.
    @param df_preprocessed: dataframe which is checked for NaNs. I insert available substitutes into the dataframe.
    @param df_inkar: (relevant parts of the) raw INKAR dataframe which possibly contains values for substituting NaNs.
    @param inds_missing: list indicators with NaNs that should be stubstituted
    @param cat_id: name of column containing ids of spatial entities
    @param cat_ind: name of column containing indicators names in INKAR dataframe
    @param cat_time: name of column containing time period inf INKAR dataframe
    @param time_period: time period of data contained in df_preprocessed
    @param cat_value: name of column containing values of indicators in INKAR dataframe
    @param spatial_type: string giving types of spatial entities: 'c' (county) or 'ma' (municipality associations)
    @return: Dataframe df_preprocessed with substitutes (when available) for NaNs of indicators given by inds_missing
    '''

    # delete csv-file with old substitute values (as there may otherwise exist files from earlier runs that are not
    # used anymore, e.g., if the indicator has been dropped)
    for filename in glob.glob(f'data/intermediate_data/INKAR_Substitutes/subs_past_{spatial_type}_*'):
        os.remove(filename)

    df_incl_substitutes = df_preprocessed.copy()
    # get dictionary of most recent substitutes
    substitutes_past = search_substitutes_from_past(df_incl_substitutes, df_inkar, inds_missing=inds_missing, cat_id=cat_id,
                                                    cat_ind=cat_ind, cat_time=cat_time, time_period=time_period,
                                                    cat_value=cat_value)
    for ind_missing in substitutes_past:
        df_substitute = substitutes_past[ind_missing]

        if not df_substitute.empty:
            df_substitute.rename({cat_value: f'{cat_value}_{ind_missing}'}, axis=1, inplace=True)
            df_substitute.to_csv(f'data/intermediate_data/INKAR_Substitutes/subs_past_{spatial_type}_{ind_missing}.csv',
                                 sep=';',index=False)
            df_incl_substitutes = df_incl_substitutes.merge(df_substitute.drop([cat_time], axis=1), how='left',
                                                            on=cat_id)
            df_incl_substitutes.loc[df_incl_substitutes[ind_missing].isna(), ind_missing] = df_incl_substitutes[
                f'{cat_value}_{ind_missing}']
            df_incl_substitutes.drop([f'{cat_value}_{ind_missing}'], axis=1, inplace=True)

    return df_incl_substitutes

def search_subs_higher_level_c(df_preprocessed, inds_missing, cat_state, cat_value):
    '''
    Computes average values of indicators given by inds_missing over all counties within the same state (omitting NaN
    values). Returns a dataframe containing the averages (indicators as columns, states as rows).
    @param df_preprocessed: Dataframe on county-level which contains the relevant data. Basis for computation of the
    averages on state-level.
    @param inds_missing: Indicators of interest with NaNs
    @param cat_state: name of column which contains ids of states
    @param cat_value: prefix of names of columns containing the state-averages in the newly generated dataframe
    @return: dataframe giving state averages of indicators (inds_missing); rows: states (1-16), columns: indicators
    with NaNs (inds_missing)
    '''
    df_substitutes = pd.DataFrame({cat_state: range(1, 17)})
    for ind in inds_missing:
        # compute mean of indicator over the counties in state
        # only compute substitutes for states with NaNs/ only substitutes that are needed
        df_substitutes.loc[df_substitutes[cat_state].isin(list(df_preprocessed.loc[df_preprocessed[ind].isna(),
        cat_state].unique())), f'{cat_value}_{ind}'] = df_substitutes[cat_state].apply(lambda state: df_preprocessed.loc[
            (df_preprocessed[cat_state] == state) & (df_preprocessed[ind].notna()), ind].mean())
    return df_substitutes


def insert_and_save_subs_higher_level_c(df_preprocessed, inds_missing, cat_state, cat_value):
    '''
    Computes state-average of indicators given by inds_missing and substitute NaNs in df_preprocessed with these
    approximations. State averages are saved in csv-file.
    @param df_preprocessed: dataframe on c-level with NaNs which are substituted
    @param inds_missing: indicators NaNs should be substituted for
    @param cat_state: name of column containing ids of states
    @param cat_value: prefix of names of columns containing the state-averages in the newly generated dataframe
    @return: dataframe on county-level with substituted NaNs. NaNs have been substituted by the averages of the
    indicator values over all counties within the same state.
    '''

    # delete csv-file with old substitutes
    for filename in glob.glob('data/intermediate_data/INKAR_Substitutes/subs_higher_level_c_*'):
        os.remove(filename)

    subs_higher_level = search_subs_higher_level_c(df_preprocessed, inds_missing=inds_missing, cat_state=cat_state,
                                                   cat_value=cat_value)
    subs_higher_level.to_csv(f'data/intermediate_data/INKAR_Substitutes/subs_higher_level_c.csv', sep=';',index=False)
    df_incl_subs = df_preprocessed.merge(subs_higher_level, how='left', on=cat_state)
    for ind in inds_missing:
        df_incl_subs.loc[df_incl_subs[ind].isna(), ind] = df_incl_subs[f'{cat_value}_{ind}']
    subs_columns = list(subs_higher_level.columns)
    subs_columns.remove(f'{cat_state}')
    df_incl_subs.drop(subs_columns, axis=1, inplace=True)
    return df_incl_subs

