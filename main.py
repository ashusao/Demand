from data import Data
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from configparser import ConfigParser
import pickle
from baseline import Baseline

if __name__ == '__main__':

    data_obj = Data()
    config = ConfigParser()
    config.read('config.ini')

    df = data_obj.read_tsv()
    baseline_approach = Baseline()

    acc_ = list()
    f1_ = list()
    cm_ = list()

    algo = config['train']['algo']
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
                n_test = X_test.shape[0]
            if algo == 'knn':
                acc, f1, cm = baseline_approach.nearest_neighbour(X_train, Y_train, X_test, Y_test, aggregate=agg_series)
            elif algo == 'rf':
                acc, f1, cm = baseline_approach.random_forest(X_train, Y_train, X_test, Y_test, aggregate=agg_series)
            acc_.append(acc)
            f1_.append(f1)
            cm_.append(cm)

    print('A: ' + str(np.mean(acc_)) + ' F1: ' + str(np.mean(f1_)))

    baseline_approach.log_result(data_type=data_type, n_train=n_train, n_test=n_test, accuracy=np.mean(acc_), f1=np.mean(f1_))









