# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os, sys, re, ast, csv, math, gc, random, enum, argparse, json, requests, time, io, base64  
from datetime import date, datetime, timedelta
from copy import deepcopy
import ast
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
import feather

from sklearn.model_selection import KFold
from catboost import Pool, CatBoostClassifier
from functools import partial  
import hyperopt
from hyperopt import hp, tpe, Trials, STATUS_OK, fmin
from sklearn.metrics import recall_score, roc_auc_score, classification_report, precision_recall_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split

import time
from contextlib import contextmanager
from dateutil.relativedelta import relativedelta


import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import string
from statsmodels.stats.outliers_influence import variance_inflation_factor    

from imblearn.under_sampling import RandomUnderSampler

from lightgbm import LGBMClassifier
import lightgbm as lgb
import matplotlib.gridspec as gridspec





def get_date_list(start_date=date(2021, 2, 1), end_date=date(2021, 11, 15)):
    date_list = [start_date + timedelta(days=i) for i in range(( end_date - start_date).days + 1)]
    return date_list


@contextmanager
def simple_timer(message='time_taken'):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print("{}: {:.3f} [s]".format(message, elapsed_time))


def reduce_mem_usage(data_df, convert_to_cat=True, verbose=True):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = data_df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in data_df.columns:
        col_type = data_df[col].dtype
        
        if col_type != object:
            c_min = data_df[col].min()
            c_max = data_df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data_df[col] = data_df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data_df[col] = data_df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data_df[col] = data_df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data_df[col] = data_df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data_df[col] = data_df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data_df[col] = data_df[col].astype(np.float32)
                else:
                    data_df[col] = data_df[col].astype(np.float64)
        else:
            if convert_to_cat:
                data_df[col] = data_df[col].astype('category')

    end_mem = data_df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        
    return data_df




def get_model_base_data(model_date, country, feature_type_list, read_csv):
    model_base_data = read_csv(f"model_base_data_{model_date.replace('-','')}.csv")
    model_base_data = model_base_data.loc[model_base_data['DerivedNationalityCountryCode'] == country, ['MemberID','target']]
    model_base_data.reset_index(drop=True, inplace=True)

    for feature_type in feature_type_list:
        feature_data = read_csv(f"model_{feature_type.lower()}_data_{model_date.replace('-','')}.csv")  
        print(f"{feature_type}: {len(feature_data)}")
        if feature_type == 'Demographic':
            cols = ['MemberID'] + ['Gender','tenure_m','age','PointsAvailableGlobal']
            feature_data = feature_data[cols]
    
            def map_gender(x):
                if pd.isna(x):
                    return -1
                else:
                    return {'M': 1, 'F': 0,}[x]
    
            feature_data['Gender'] = feature_data['Gender'].apply(lambda x: map_gender(x)).astype(int)
        feature_data = reduce_mem_usage(feature_data)
        model_base_data = pd.merge(model_base_data, feature_data, how='left', on='MemberID')
     
    return model_base_data



def process_feature_data(data_df, feature_type):

    if feature_type == 'Demographic':
        cols = ['MemberID'] + ['Gender','tenure_months','age_years','PointsAvailableGlobal']
        data_df = data_df[cols]

        def map_gender(x):
            if pd.isna(x):
                return -1
            else:
                return {'M': 1, 'F': 0,}[x]

        data_df['Gender'] = data_df['Gender'].apply(lambda x: map_gender(x)).astype(int)
        
    data_df = reduce_mem_usage(data_df)
    
    return data_df



def get_model_data(model_date, country, feature_type_list, col_id, col_target, dataPath):
    file_date = model_date.replace('-','')
    model_data = feather.read_dataframe(dataPath / f"model_base_data_{country}_{file_date}")
    model_data = model_data[[col_id, col_target]].copy()    

    for feature_type in feature_type_list:
        feature_data = feather.read_dataframe(dataPath / f"model_{feature_type.lower()}_data_{country}_{file_date}")  
        model_data = pd.merge(model_data, feature_data, how='left', on=col_id)
    
    return model_data


def get_model_date(test_model_date): 
    train_model_date = str(pd.to_datetime(test_model_date) + relativedelta(months=-2)).split(' ')[0]
    valid_model_date = str(pd.to_datetime(test_model_date) + relativedelta(months=-1)).split(' ')[0]
    return (train_model_date, valid_model_date)




def get_classifier_model(data_df, col_target, col_num, col_cat, params=None):
    if params is None:
        model = CatBoostClassifier(
                                   random_seed=100,
                                   od_type='Iter', od_wait=20, 
                                   eval_metric='AUC', 
                                   verbose = 0,                                                                 
                                   fold_len_multiplier=2,   
                                   allow_writing_files=False,
                                   )   
    else:
        model = CatBoostClassifier(
                                   random_seed=params['model_seed'],
                                   od_type='Iter', od_wait=20, 
                                   eval_metric=params['eval_metric'], 
                                   verbose = 0,                                                                 
                                   fold_len_multiplier=params['fold_len_multiplier'],
                                   learning_rate=params['learning_rate'],
                                   depth=int(params['depth']),
                                   l2_leaf_reg=params['l2_leaf_reg'],   
                                   allow_writing_files=False,
                                   )      

    model.fit(data_df[col_num+col_cat], data_df[col_target], col_cat)
    return model





def decile_analysis(data_df, col_id, col_target, y_prob, y_true):
    decile_df = data_df[[col_id, col_target]].copy()
    decile_df['y'] = y_true
    decile_df['y_prob'] = y_prob
    
    base_response_rate = np.round(100*decile_df[decile_df.y == 1].shape[0]/decile_df.shape[0], decimals = 2)
    decile_df.sort_values(by = 'y_prob', inplace = True, ascending = False)
    decile_df.reset_index(inplace = True)
    decile_df['decile'] = np.nan
    d = int(np.ceil(decile_df.shape[0]/10))
    start = 0
    end = d
    
    for i in range(10):
        decile_df.loc[start:end, ['decile']] = i + 1
        start = start + d
        end = end + d
        
    decile_result_df = pd.crosstab(decile_df['decile'], decile_df['y'])
    decile_result_df.columns = ['zero', 'one']
    decile_result_df['min_prob'] = decile_df.groupby(by = ['decile']).min()['y_prob']
    decile_result_df['max_prob'] = decile_df.groupby(by = ['decile']).max()['y_prob']
    decile_result_df['count'] = decile_df.groupby(by = ['decile']).count()['y_prob']
    decile_result_df['gain'] = np.round(100*decile_result_df['one']/decile_result_df['one'].sum(), decimals = 2)
    decile_result_df['cum_gain'] = np.cumsum(decile_result_df['gain'])
    decile_result_df['lift'] = np.round((100*decile_result_df['one']/decile_result_df['count'])/base_response_rate, 2)  
    
    return decile_result_df




def lift_analysis(data_df, col_id, col_target, y_prob, y_true):
    decile_df = data_df[[col_id, col_target]].copy()
    decile_df['y'] = y_true
    decile_df['y_prob'] = y_prob
    
    base_response_rate = np.round(100*decile_df[decile_df.y == 1].shape[0]/decile_df.shape[0], decimals = 2)
    decile_df.sort_values(by = 'y_prob', inplace = True, ascending = False)
    decile_df.reset_index(inplace = True)
    decile_df['group'] = np.nan
    d = int(np.ceil(decile_df.shape[0]/5))
    start = 0
    end = d
    
    for i in range(5):
        decile_df.loc[start:end, ['group']] = i + 1
        start = start + d
        end = end + d
        
    lift_result_df = pd.crosstab(decile_df['group'], decile_df['y'])
    lift_result_df.columns = ['zero', 'one']
    lift_result_df['min_prob'] = decile_df.groupby(by = ['group']).min()['y_prob']
    lift_result_df['max_prob'] = decile_df.groupby(by = ['group']).max()['y_prob']
    lift_result_df['count'] = decile_df.groupby(by = ['group']).count()['y_prob']
    lift_result_df['gain'] = np.round(100*lift_result_df['one']/lift_result_df['one'].sum(), decimals = 2)
    lift_result_df['cum_gain'] = np.cumsum(lift_result_df['gain'])
    lift_result_df['lift'] = np.round((100*lift_result_df['one']/lift_result_df['count'])/base_response_rate, 2)
    
    return lift_result_df





# define a binning function
def mono_bin(Y, X, n=20, force_bin=3):
    
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)

def char_bin(Y, X):
        
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]    
    df2 = notmiss.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)


def get_iv_df(data_df, col_target, col_num):

    count = -1
    for col in col_num:
        
        if len(pd.Series.unique(data_df[col])) > 1:
            conv = mono_bin(data_df[col_target], data_df[col])
            conv["VAR_NAME"] = col
   
        count = count + 1

        if count == 0:
            iv_df = conv
        else:
            iv_df = iv_df.append(conv,ignore_index=True)

    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return (iv_df, iv)






def fill_na_df(data_df, col_num, col_cat, na_val_num=0, na_val_cat='NA'):
    data_df[col_num] = data_df[col_num].fillna(na_val_num)
    data_df[col_cat] = data_df[col_cat].fillna(na_val_cat)    
    return data_df


# =============================================================================
# def get_col_features(data_df, col_id, col_target, na_pct_thres=1.0):
#     data_df_copy = data_df.copy()
#
#     col_num = [c for c in data_df_copy.columns if c not in [col_id, col_target]]
#     col_cat = []
#     
#     df_temp = data_df_copy[col_num+col_cat].isna().mean()
#     cols_to_rm = list(df_temp[(df_temp >= na_pct_thres)].index)
#     col_num = [c for c in col_num if c not in cols_to_rm]
#     col_cat = [c for c in col_cat if c not in cols_to_rm]
#     data_df_copy = fill_na_df(data_df_copy, col_num, col_cat)
#     
#     df_temp = data_df_copy[col_num].quantile(0.95) - data_df_copy[col_num].quantile(0.05)
#     cols_to_rm = list(df_temp[df_temp == 0].index)
#     col_num = [c for c in col_num if c not in cols_to_rm]
#
#     return (col_num, col_cat)
# =============================================================================


def get_col_features(data_df, col_id, col_target):
    col_num = [c for c in data_df.columns if c not in [col_id, col_target]]
    col_cat = []
    
    cols_to_rm = []
    for col in col_num:
        if len(Series.unique(data_df[col])) <= 1:
            cols_to_rm += [col]
    
    col_num = [c for c in col_num if c not in cols_to_rm]    

    return (col_num, col_cat)






def get_col_num_vif(data_df, col_num, thresh=5):
    col_num_arr = np.array(col_num)
    col_idx = np.arange(len(col_num_arr))
    dropped=True
    while dropped:
        dropped=False
        c = data_df[col_num_arr[col_idx]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            col_idx = np.delete(col_idx, maxloc)
            dropped=True

    col_num_vif = list(col_num_arr[col_idx])
    return col_num_vif





def get_feature_importances(X, y, col_features, shuffle=False, seed=None):
    y_copy = y.copy()
    if shuffle:
        y_copy = y_copy.sample(frac=1.0)
    
    if seed is not None:
        np.random.seed(seed)

    dtrain = None
    dtrain = lgb.Dataset(X[col_features], y_copy, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 127,
        'max_depth': 8,
        'seed': seed,
        'bagging_freq': 1,
        'n_jobs': 4,
        'verbose': -1,
    }
    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200, verbose_eval=False)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(col_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y_copy, clf.predict(X[col_features]))
    
    return imp_df


def get_null_imp_df(nb_runs, X, y, col_features):
    null_imp_df = pd.DataFrame()
    start = time.time()
    for i in range(nb_runs):
        imp_df = get_feature_importances(X, y, col_features, shuffle=True, seed=i)
        imp_df['run'] = i + 1 
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)      
    return null_imp_df



def get_feature_scores(actual_imp_df, null_imp_df):
    feature_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
        gain_score = np.log(1e-10 + (1 + f_act_imps_gain) / (1 + np.percentile(f_null_imps_gain, 95)))  # Avoid didvide by zero
        f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
        split_score = np.log(1e-10 + (1 + f_act_imps_split) / (1 + np.percentile(f_null_imps_split, 95)))  # Avoid didvide by zero
        feature_scores.append((_f, split_score, gain_score))
    
    scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
    scores_df.sort_values('split_score', ascending=False, inplace=True)
    scores_df.reset_index(drop=True, inplace=True)
    return scores_df





def get_correlation_scores(actual_imp_df, null_imp_df):
    correlation_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
        gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
        split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        correlation_scores.append((_f, split_score, gain_score))
    corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
    return corr_scores_df


def get_important_features(X, y, col_features, nb_runs=20, score_type='split_score', show_plot=False):
    actual_imp_df = get_feature_importances(X, y, col_features, shuffle=False, seed=100)
    null_imp_df = get_null_imp_df(nb_runs, X, y, col_features)
    scores_df = get_feature_scores(actual_imp_df, null_imp_df)
    features = list(scores_df.loc[scores_df[score_type] > 0, 'feature'])
    
    if show_plot:
        plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(1, 2)
        # Plot Split importances
        ax = plt.subplot(gs[0, 0])
        sns.barplot(x='split_score', y='feature', data=scores_df.sort_values('split_score', ascending=False).iloc[0:70], ax=ax)
        ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
        # Plot Gain importances
        ax = plt.subplot(gs[0, 1])
        sns.barplot(x='gain_score', y='feature', data=scores_df.sort_values('gain_score', ascending=False).iloc[0:70], ax=ax)
        ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
        plt.tight_layout()
    return features




def get_bayes_opt_params(df_train, df_valid, col_id, col_target, col_num, col_cat):

    def bayes_objective(params):
        model = get_classifier_model(df_train, col_target, col_num, col_cat, params)                  
        y_prob = model.predict_proba(df_valid)[:,1]
        y_valid = df_valid[col_target].values
        decile_result_df = decile_analysis(df_valid, col_id, col_target, y_prob, y_valid)
        score = decile_result_df.loc[:4, 'lift'].sum()
        loss = 1 - score
        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    space = {  
            'eval_metric': hp.choice('eval_metric', ['AUC']),                              
            'fold_len_multiplier': 2,  
            'model_seed': 100,     
            'depth': hyperopt.hp.choice('depth', np.arange(3, 11)),
            'learning_rate': hyperopt.hp.uniform('learning_rate', 0.01, 0.2),        
            'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 8),               
            }         

    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()
    bayes_output = fmin(fn = bayes_objective, 
                        space = space, 
                        algo = partial(tpe.suggest, n_startup_jobs = 5, n_EI_candidates = 12), 
                        max_evals = 50,
                        trials = bayes_trials)
    
    bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
    best_bayes_output = bayes_trials_results[0]
    
    return best_bayes_output




def get_undersampled_data(data_df, col_num, col_target):
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=100)    
    df_rus, y_rus = rus.fit_resample(data_df[col_num], data_df[col_target])  
    df_rus[col_target] = y_rus.values
    return df_rus        


def get_rfe_features(data_df, col_num, col_target, n_feat=50, undersample=False):
    if undersample:
        data_df = get_undersampled_data(data_df, col_num, col_target) 
        
    model = CatBoostClassifier(
                               random_seed=100,
                               od_type='Iter', od_wait=20, 
                               eval_metric='AUC', 
                               verbose = 0,                                                                 
                               fold_len_multiplier=2,   
                               allow_writing_files=False,
                               )      
    
    output = model.select_features(data_df[col_num], data_df[col_target], 
                                   features_for_select=col_num,
                                   num_features_to_select=n_feat)         

    return output['selected_features_names']    



def get_model_date_list(initial_date):
    model_date_list = [initial_date + timedelta(days=x) for x in range((datetime.date(datetime.now())-initial_date).days + 1)]
    model_date_list = [str(d).split(' ')[0] for d in model_date_list if d.day==1]
    return model_date_list






