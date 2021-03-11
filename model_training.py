import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
import gc
import lightgbm as lgb
import shap
from hyperopt import hp
from hyperopt import Trials
from hyperopt import tpe
from hyperopt import fmin
from hyperopt import STATUS_OK

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from math import sqrt


def comp_auc(target, predictions):
    """
    computes auc
    """
    return roc_auc_score(target, predictions)


def com_rsq(target, predictions):
    """
    computes r-squared
    """
    return sqrt(mean_squared_error(target, predictions))


def one_model(x_train, x_valid, y_train, y_valid, params):
    """
    Trains 1 LGBM model 
    """

    dtrain=lgb.Dataset(x_train, label=y_train)
    dvalid=lgb.Dataset(x_valid, label=y_valid)

    watchlist=dvalid

    booster=lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=watchlist,
        verbose_eval=200
    )

    return booster

def fit_model(x_train,y_train,params,CV_folds, preds): # fits Cross Validation
    """
    Trains LGBM cross validation 
    """

    folds=KFold(n_splits=CV_folds)
    models=[]
    performance = []

    for train_index, valid_index in folds.split(x_train):
        X_train, X_valid = x_train.loc[train_index,:], x_train.loc[valid_index, :] 
        Y_train, Y_valid = y_train.loc[train_index], y_train.loc[valid_index]



        model = one_model(X_train[preds], X_valid[preds], Y_train, Y_valid,params)
        models.append(model)

        x=[]
        x.append(model)
        predictions = predict(x, X_valid, preds)

        if (Y_valid.nunique() == 2):     # classification
            performance.append(comp_auc(Y_valid, predictions))
        
        else:
            performance.append(com_rsq(Y_valid, predictions))


    print('Performance on validation sets:')
    print(performance)
    print('Mean:')
    print(np.mean(performance))

    return models


def predict(models, set_to_predict,preds):
    """
    Predicts average results from list of models
    """
    
    predictions=np.zeros(set_to_predict.shape[0])

    for model in models:
        predictions = predictions + model.predict(set_to_predict[preds])/len(models)

    return predictions

def comp_var_imp(models,preds):
    """
    Computes variable importance of lgbm model(s)
    """

    importance_df=pd.DataFrame()
    importance_df['Feature']=preds
    importance_df['Importance_gain']=0
    importance_df['Importance_weight']=0

    for model in models:
        importance_df['Importance_gain'] = importance_df['Importance_gain'] + model.feature_importance(importance_type = 'gain') / len(models)
        importance_df['Importance_weight'] = importance_df['Importance_weight'] + model.feature_importance(importance_type = 'split') / len(models)

    return importance_df

def plot_importance(models, imp_type, preds ,ret=False, show=True, n_predictors = 100):
    """
    Plots variable importances
    """
    if ((imp_type!= 'Importance_gain' ) & (imp_type != 'Importance_weight')):
        raise ValueError('Only importance_gain or importance_gain is accepted')

    dataframe = comp_var_imp(models, preds)

    if (show == True):
        plt.figure(figsize = (20, len(preds)/2))
        sns.barplot(x=imp_type, y='Feature', data=dataframe.sort_values(by=imp_type, ascending= False).head(len(preds)))

    if (ret == True):
        return dataframe.sort_values(by=imp_type, ascending= False).head(len(preds))[['Feature', imp_type]]




def param_hyperopt(params, x_train,  y_train, n_iter = 500):
    """
    Runs hyperparameter optimization
    """

    train_data = lgb.Dataset(x_train, label = y_train)

    def objective (params):

        print(params)

        if (y_train.nunique()==2):
                             
            cv_results = lgb.cv(params, train_data, stratified = True, nfold = 4)

            best_score = -max(cv_results['auc-mean'])
            print('----------------------------------------------------------------------------------')

            return {'loss': best_score, 'params': params, 'status': STATUS_OK}

        else:
            cv_results = lgb.cv(params, train_data, nfold = 4)
            best_score = -max(cv_results['rmse-mean'])
            print('----------------------------------------------------------------------------------')
            return {'loss': best_score, 'params': params, 'status': STATUS_OK}

    space = {
            'learning_rate': hp.choice('learning_rate', np.arange(0.02, 0.1, 0.02, dtype=float)),
            'num_leaves': hp.choice('num_leaves', np.arange(2, 64, 2, dtype=int)),
            'max_depth': hp.choice('max_depth', np.arange(2, 5, 1, dtype=int)),
            'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.5, 0.9, 0.05, dtype=float)),
            'subsample': hp.choice('subsample', np.arange(0.5, 0.9, 0.05, dtype=float)),
            'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(50, 500, 50, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(10, 100, 10, dtype=int)),
            'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            'verbose':1,
            'metric':'auc',
            'objective':'binary',
            'early_stopping_rounds':50,
            'num_boost_round':100000,
            'seed':1234
            }


    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()

    best = fmin(fn = objective, space = space, algo = tpe_algorithm, 
    max_evals = n_iter, trials = bayes_trials)

    print('Best combination of parameters is:')
    print(best)
    print('')
    return best


def print_shap_values(preds, cols_num, cols_cat, x_train, y_train, params):
    """
    Computes shap values of lgbm model
    """
    x_train = x_train[preds]


    for col in cols_cat:
        x_train[col] = x_train[col].cat.add_categories('NA').fillna('NA')
        _ , indexer = pd.factorize(x_train[col])
        x_train[col] = indexer.get_indexer(x_train[col])

    model=one_model(x_train, x_train, y_train, y_train, params)
        
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_train)

    if isinstance(shap_values, list):
               
        shap_values = shap_values[1]

    else:
        shap_values = shap_values


    shap.summary_plot(shap_values, x_train)
    shap.summary_plot(shap_values, x_train, plot_type='bar')
 
    var_imp_dataframe = {'Feature': preds, 'Shap_importance': np.mean(abs(shap_values),axis=0) }

    return pd.DataFrame(var_imp_dataframe).sort_values(by=['Shap_importance'], ascending = False), shap_values ,explainer
        

        

def print_shap_interaction_matrix(preds, set_to_shap, explainer):
    """
    Computes matrix of shap interaction values and prints it
    """

    shap_inter_values = explainer.shap_interaction_values(set_to_shap)

    corr = np.sum(abs(shap_inter_values),axis=0)
    corr[np.diag_indices_from(corr)] = 0

    corr = pd.DataFrame(corr, columns=preds, index=preds)
    corr = round(corr,0)

    sns.set(rc={'figure.figsize':(20,20)})
    sns.heatmap(corr, annot=True, xticklabels=preds, yticklabels=preds, fmt='.5g')


def shap_dependence_plot(x_train, cols_cat, shap_values, x, y=None):
    """
    Plots shap dependence plot for 1 variable
    """
    if x in cols_cat:         
        print('Encoding of categories for your variable')   
        labels, uniques = pd.factorize(x_train[x].cat.add_categories('NA').fillna('NA'))
        for j in range(len(np.unique(labels))):
            print(np.unique(labels)[j])
            print(uniques[j])


    if  y is None:
        shap.dependence_plot(x, shap_values, x_train)

    else:            
        shap.dependence_plot(x, shap_values, x_train, interaction_index = y)
