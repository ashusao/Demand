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
from test_data import generate_test_set
from test_data import evaluate_test_set
from mlsmote import apply_mlsmote
from dd_data import read_and_convert_dd
from dd_data import split_train_test_
from dd_test_data import generate_test_set_
import json

if __name__ == '__main__':

    data_obj = Data()
    config = ConfigParser()
    config.read('config.ini')
    dataset = config['data']['dataset']

    if dataset == 'demand':
        df = read_and_convert_dd(config, 'dataset.xlsx', config['data']['train_start'], config['data']['val_stop'])
    else:
        df = data_obj.read_tsv('aug_dec_no_filter.tsv', config['data']['train_start'], config['data']['val_stop'])
    print(df.shape)

    feat = config.getboolean('data', 'features')
    eval_train = config.getboolean('data', 'eval_train')
    train_ = config.getboolean('data', 'train')
    eval_tests = config.getboolean('data', 'eval_tests')
    oversample = config.getboolean('data', 'oversample')
    n_sample = int(config['data']['n_sample'])
    algo = config['train']['algo']

    if algo == 'knn' or algo == 'rf' or algo == 'ha' or algo == 'lr':
        baseline_approach = Baseline()

    horizons = config.get("data", "input_horizons")
    ip_horizons = json.loads(horizons)

    for ip_horizon in ip_horizons:
        ip_horizon = int(ip_horizon)
        if feat:
            if dataset == 'demand':
                X_train, Y_train, X_test, Y_test, \
                Train_pattern_features, Test_pattern_features = split_train_test_(config, df)
                Train_cs_features = np.random.rand(X_train.shape[0], 2)
                Test_cs_features = np.random.rand(X_test.shape[0], 2)
                Train_spatial_features = np.random.rand(X_train.shape[0], 2)
                Test_spatial_features = np.random.rand(X_test.shape[0], 2)
            else:
                X_train, Y_train, Train_cs_features, Train_spatial_features, Train_pattern_features, \
                Train_median_features, Train_q25_features, Train_q75_features = \
                    data_obj.split_train_test(df, ip_horizon)
        else:
            if dataset == 'demand':
                X_train, Y_train, X_test, Y_test = split_train_test_(config, df)
            else:
                X_train, Y_train = data_obj.split_train_test(df, ip_horizon)
            Train_cs_features = np.random.rand(len(X_train), 2)
            Train_spatial_features = np.random.rand(len(X_train), 2)
            Train_pattern_features = np.random.rand(len(X_train), 2)
            Train_median_features = np.random.rand(len(X_train), 2)
            Train_q25_features = np.random.rand(len(X_train), 2)
            Train_q75_features = np.random.rand(len(X_train), 2)

        if oversample:
            X_train, Y_train, Train_cs_features, Train_spatial_features = \
                apply_mlsmote(config, X_train, Y_train, Train_cs_features, Train_spatial_features, n_sample)

        if train_:

            if algo == 'knn':
                baseline_approach.nearest_neighbour(X_train, Y_train, ip_horizon)
            elif algo == 'rf':
                baseline_approach.random_forest(X_train, Y_train, ip_horizon)
            elif algo == 'svm':
                baseline_approach.support_vector_classifier(X_train, Y_train, ip_horizon)
            elif algo == 'ha':
                pass
            else:
                train(config, X_train, Y_train, Train_cs_features, Train_spatial_features, Train_pattern_features,
                      Train_median_features, Train_q25_features, Train_q75_features, ip_horizon)

        if eval_train:
            evaluate(config, X_train, Y_train, Train_cs_features, Train_spatial_features, Train_pattern_features, X_train.shape[0], ip_horizon)

        if eval_tests:
            if feat:
                if dataset == 'demand':
                    X, Y, Feat_pattern = generate_test_set_(config)
                    Feat_cs = [np.random.rand(X[0].shape[0], 2)] * 4
                    Feat_spatial = [np.random.rand(X[0].shape[0], 2)] * 4
                else:
                    X, Y, Feat_cs, Feat_spatial, Feat_pattern, Feat_median, Feat_q25, Feat_q75 = \
                        generate_test_set(config, ip_horizon)
            else:
                if dataset == 'demand':
                    X, Y = generate_test_set_(config)
                else:
                    X, Y = generate_test_set(config, ip_horizon)
                Feat_cs = [np.random.rand(len(X[0]), 2)] * 5
                Feat_spatial = [np.random.rand(len(X[0]), 2)] * 5
                Feat_pattern = [np.random.rand(len(X[0]), 2)] * 5
                Feat_median = [np.random.rand(len(X[0]), 2)] * 5
                Feat_q25 = [np.random.rand(len(X[0]), 2)] * 5
                Feat_q75 = [np.random.rand(len(X[0]), 2)] * 5

            if algo == 'knn' or algo == 'rf' or algo == 'ha' or algo == 'lr':
                baseline_approach.eval_test_set(len(X_train), X, Y, ip_horizon)
            else:
                evaluate_test_set(config, X, Y, Feat_cs, Feat_spatial, Feat_pattern, Feat_median, Feat_q25,
                                  Feat_q75, len(X_train), ip_horizon)














