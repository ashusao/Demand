# -*- coding: utf-8 -*-
# Importing required Library
import numpy as np
import pandas as pd
import os
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors

def create_dataset(n_sample=1000):
    ''' 
    Create a unevenly distributed sample data set multilabel  
    classification using make_classification function
    
    args
    nsample: int, Number of sample to be created
    
    return
    X: pandas.DataFrame, feature vector dataframe with 10 features 
    y: pandas.DataFrame, target vector dataframe with 5 labels
    '''
    X, y = make_classification(n_classes=5, class_sep=2, 
                           weights=[0.1,0.025, 0.205, 0.008, 0.9], n_informative=3, n_redundant=1, flip_y=0,
                           n_features=10, n_clusters_per_class=1, n_samples=1000, random_state=10)
    y = pd.get_dummies(y, prefix='class')
    return pd.DataFrame(X), y


def get_tail_label(df):
    """
    Give tail label colums of the given target dataframe
    
    args
    df: pandas.DataFrame, target label df whose tail label has to identified
    
    return
    tail_label: list, a list containing column name of all the tail label
    """
    columns = df.columns
    n = len(columns)
    irpl = np.zeros(n)
    for column in range(n):
        irpl[column] = df[columns[column]].value_counts()[1]
    irpl = max(irpl)/irpl
    mir = np.average(irpl)
    tail_label = []
    for i in range(n):
        if irpl[i] > mir:
            tail_label.append(columns[i])
    return tail_label


def get_index(df):
  """
  give the index of all tail_label rows
  args
  df: pandas.DataFrame, target label df from which index for tail label has to identified
    
  return
  index: list, a list containing index number of all the tail label
  """
  tail_labels = get_tail_label(df)
  index = set()
  for tail_label in tail_labels:
    sub_index = set(df[df[tail_label]==1].index)
    index = index.union(sub_index)
  return list(index)


def get_minority_instance(config, X, y, cs_feat, spatial_feat):
    """
    Give minority dataframe containing all the tail labels
    
    args
    X: pandas.DataFrame, the feature vector dataframe
    y: pandas.DataFrame, the target vector dataframe
    
    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    index = get_index(y)
    X_sub = X[X.index.isin(index)].reset_index(drop = True)
    y_sub = y[y.index.isin(index)].reset_index(drop = True)

    feat = config.getboolean('data', 'features')
    if feat:
        cs_sub = cs_feat[cs_feat.index.isin(index)].reset_index(drop = True)
        spatial_sub = spatial_feat[spatial_feat.index.isin(index)].reset_index(drop = True)
        return X_sub, y_sub, cs_sub, spatial_sub
    else:
        return X_sub, y_sub


def get_minority_instance_new(config, X, y, cs_feat, spatial_feat):

    y['minority_prop'] = y.sum(axis=1) / y.shape[1]
    mean_prop = np.mean(y['minority_prop'])
    index = y[y['minority_prop'] > mean_prop].index
    y.drop(['minority_prop'], axis=1, inplace=True)

    X_sub = X[X.index.isin(index)].reset_index(drop=True)
    y_sub = y[y.index.isin(index)].reset_index(drop=True)

    feat = config.getboolean('data', 'features')
    if feat:
        cs_sub = cs_feat[cs_feat.index.isin(index)].reset_index(drop=True)
        spatial_sub = spatial_feat[spatial_feat.index.isin(index)].reset_index(drop=True)
        return X_sub, y_sub, cs_sub, spatial_sub
    else:
        return X_sub, y_sub


def nearest_neighbour(X):
    """
    Give index of 5 nearest neighbor of all the instance
    
    args
    X: np.array, array whose nearest neighbor has to find
    
    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs=NearestNeighbors(n_neighbors=5,metric='euclidean',algorithm='kd_tree').fit(X)
    euclidean,indices= nbs.kneighbors(X)
    return indices

def MLSMOTE(feat, X,y, cs, spatial, n_sample):
    """
    Give the augmented data using MLSMOTE algorithm
    
    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample
    
    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    indices2 = nearest_neighbour(X)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    cs_feat = np.zeros((n_sample, cs.shape[1]))
    spatial_feat = np.zeros((n_sample, spatial.shape[1]))

    for i in range(n_sample):
        reference = random.randint(0, n-1)
        neighbour = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis = 0, skipna = True)
        target[i] = np.array([1 if val>2 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference,:] - X.loc[neighbour,:]
        new_X[i] = np.array(X.loc[reference,:] + ratio * gap)
        #new_X[i] = X.loc[neighbour, :]
        #target[i] = y.loc[neighbour, :]
        cs_feat[i] = cs.loc[neighbour, :]
        spatial_feat[i] = spatial.loc[neighbour, :]
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    #new_X = pd.concat([X, new_X], axis=0)
    #target = pd.concat([y, target], axis=0)
    cs_feat = pd.DataFrame(cs_feat, columns=cs.columns)
    spatial_feat = pd.DataFrame(spatial_feat, columns=spatial.columns)
    return new_X, target, cs_feat, spatial_feat

def apply_mlsmote(config, X, Y, cs_feat, spatial_feat, n_sample):

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    cs_feat = pd.DataFrame(cs_feat)
    spatial_feat = pd.DataFrame(spatial_feat)

    feat = config.getboolean('data', 'features')
    mlsmote_path = config['data']['mlsmote_path']
    input_horizon = int(config['data']['input_horizon'])

    # generate and save npy
    if not os.path.isfile(os.path.join(mlsmote_path, 'X_lag_' + str(input_horizon) + '.npy')):
        if feat:
            x_sub, y_sub, cs_sub, spatial_sub = get_minority_instance(config, X, Y, cs_feat, spatial_feat)
            print('Minority Instance: ')
            print(x_sub.shape, y_sub.shape, cs_sub.shape, spatial_sub.shape)
            x_res, y_res, cs_res, spatial_res = MLSMOTE(feat, x_sub, y_sub, cs_sub, spatial_sub, n_sample)
            print('Generated Instance: ')
            print(x_res.shape, y_res.shape, cs_res.shape, spatial_res.shape)
        else:
            x_sub, y_sub = get_minority_instance(config, X, Y, cs_feat, spatial_feat)
            print('Minority Instance: ')
            print(x_sub.shape, y_sub.shape)
            x_res, y_res, cs_res, spatial_res = MLSMOTE(feat, x_sub, y_sub, cs_feat, spatial_feat, n_sample)
            print('Generated Instance: ')
            print(x_res.shape, y_res.shape, cs_res.shape, spatial_res.shape)

        X = np.vstack((X, x_res))
        Y = np.vstack((Y, y_res))
        cs = np.vstack((cs_feat, cs_res))
        spatial = np.vstack((spatial_feat, spatial_res))

        np.save(os.path.join(mlsmote_path, 'X_lag_' + str(input_horizon) + '.npy'), X)
        np.save(os.path.join(mlsmote_path, 'Y_lag_' + str(input_horizon) + '.npy'), Y)
        np.save(os.path.join(mlsmote_path, 'cs_lag_' + str(input_horizon) + '.npy'), cs)
        np.save(os.path.join(mlsmote_path, 'spatial_lag_' + str(input_horizon) + '.npy'), spatial)

    X_new = np.load(os.path.join(mlsmote_path, 'X_lag_' + str(input_horizon) + '.npy'))
    Y_new = np.load(os.path.join(mlsmote_path, 'Y_lag_' + str(input_horizon) + '.npy'))
    cs_new = np.load(os.path.join(mlsmote_path, 'cs_lag_' + str(input_horizon) + '.npy'))
    spatial_new = np.load(os.path.join(mlsmote_path, 'spatial_lag_' + str(input_horizon) + '.npy'))

    return X_new, Y_new, cs_new, spatial_new



