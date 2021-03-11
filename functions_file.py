import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
import gc
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit


def detect_types(data):
    """
    Separates columns to numerical ones and categorical ones
    """
    numerical_preds=[]
    categorical_preds=[]
     
    
    for i in list(data):
        if(data[i].dtype=='object'):
            categorical_preds.append(i)
        else:
            numerical_preds.append(i)
    
    return numerical_preds,categorical_preds




def graph_exploration(feature_binned,target):
    """
    Function that visualises relationship between given binned variable and 
    binary target
    """
  
    if(sum(feature_binned.isnull())>0):
        feature_binned=feature_binned.cat.add_categories('NA').fillna('NA')
    
    result = pd.concat([feature_binned, target], axis=1)
    
    gb=result.groupby(feature_binned)
    counts = gb.size().to_frame(name='counts')
    final=counts.join(gb.agg({result.columns[1]: 'mean'}).rename(columns={result.columns[1]: 'target_mean'})).reset_index()
    final['pom.sanci']=np.log2((final['counts']*final['target_mean']+100*np.mean(target))/((100+final['counts'])*np.mean(target)))
        
    sns.set(rc={'figure.figsize':(15,10)})
    fig, ax =plt.subplots(2,1)
    sns.countplot(x=feature_binned, hue=target, data=result,ax=ax[0])
    sns.barplot(x=final.columns[0],y='pom.sanci',data=final,color="green",ax=ax[1])
    plt.show()
    
    
def graph_exploration_continuous(feature_binned,target):
    """
    Function that visualises relationship between given binned variable and 
    continuous target
    """
  
    if(sum(feature_binned.isnull())>0):
        feature_binned=feature_binned.cat.add_categories('NA').fillna('NA')
    
    plt.figure(figsize=(12,5))
    sns.boxplot(x=feature_binned,y=target,showfliers=False)
    plt.xticks(rotation='vertical')
    #plt.xlabel(feature_binned, fontsize=12)
    #plt.ylabel(target, fontsize=12)
    plt.show()
    
    
    
    
def replace_categories(train_set,test_set,categorical_preds,num_categories):   
    """
    Merges categories of variables with more than num_categories categories
    and predits this transformation to test data
    """
    for i in categorical_preds:
        if train_set[i].nunique()>num_categories:
            print(i)
            print(train_set[i].nunique())
            top_n_cat=train_set[i].value_counts()[:10].index.tolist()
            train_set[i]=np.where(train_set[i].isin(top_n_cat),train_set[i],'other')   
            test_set[i]=np.where(test_set[i].isin(top_n_cat),test_set[i],'other')
    return train_set,test_set



def reduce_mem_usage(df):
    """
    Downloaded function. 
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

    
    