from data import Data
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from configparser import ConfigParser
import pickle
from baseline import Baseline
from train import train
from train import evaluate
from train import log_result

if __name__ == '__main__':

    data_obj = Data()
    config = ConfigParser()
    config.read('config.ini')

    df = data_obj.read_tsv()
    baseline_approach = Baseline()

    acc_ = list()
    f1_ = list()
    cm_ = list()

    '''algo = config['train']['algo']
    agg_series = config.getboolean('data', 'complete_data')
    if agg_series:
        data_type = 'Complete'
    else:
        data_type = 'Individual'

    # For complete data
    n_train = 0
    n_test = 0

    if agg_series:
        X_train, Y_train, X_test, Y_test = data_obj.split_train_test(df, 0, aggregate=agg_series)
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        if algo == 'knn':
            acc, f1, cm = baseline_approach.nearest_neighbour(X_train, Y_train, X_test, Y_test, aggregate=agg_series)
        elif algo == 'rf':
            acc, f1, cm = baseline_approach.random_forest(X_train, Y_train, X_test, Y_test, aggregate=agg_series)
        acc_.append(acc)
        f1_.append(f1)
        cm_.append(cm)
    else:
        for idx in range(df.shape[1]):
            X_train, Y_train, X_test, Y_test = data_obj.split_train_test(df, idx, aggregate=agg_series)
            if idx == 0:
                n_train = X_train.shape[0]
                n_test = X_test.shape[0]ls 
            if algo == 'knn':
                acc, f1, cm = baseline_approach.nearest_neighbour(X_train, Y_train, X_test, Y_test, aggregate=agg_series)
            elif algo == 'rf':
                acc, f1, cm = baseline_approach.random_forest(X_train, Y_train, X_test, Y_test, aggregate=agg_series)
            acc_.append(acc)
            f1_.append(f1)
            cm_.append(cm)

    print('A: ' + str(np.mean(acc_)) + ' F1: ' + str(np.mean(f1_, axis=0)))

    baseline_approach.log_result(data_type=data_type, n_train=n_train, n_test=n_test, accuracy=np.mean(acc_), f1=np.mean(f1_, axis=0))'''

    #X_train, Y_train, X_test, Y_test, Train_features, Test_features = data_obj.split_train_test(df, 0, aggregate=True)
    Train_features = np.array([1, 2, 3, 4, 5])
    Test_features = np.array([1, 2, 3, 4, 5])
    X_train, Y_train, X_test, Y_test= data_obj.split_train_test(df, 0, aggregate=True)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, Train_features.shape, Test_features.shape)

    train(config, X_train, Y_train, X_test, Y_test, Train_features, Test_features)

    evaluate(config, X_test, Y_test, Test_features, X_train.shape[0])
    #evaluate(config, X_train, Y_train, Train_features, X_train.shape[0])









