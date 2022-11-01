from data import Data
import numpy as np
from configparser import ConfigParser
from baseline import Baseline
from train import train
from train import evaluate
from test_data import generate_test_set
from test_data import evaluate_test_set
import json

if __name__ == '__main__':

    data_obj = Data()
    config = ConfigParser()
    config.read('config.ini')
    dataset = config['data']['dataset']

    df = data_obj.read_tsv('aug_dec_no_filter.tsv', config['data']['train_start'], config['data']['val_stop'])
    print(df.shape)

    feat = config.getboolean('data', 'features')
    eval_train = config.getboolean('data', 'eval_train')
    train_ = config.getboolean('data', 'train')
    eval_tests = config.getboolean('data', 'eval_tests')
    #oversample = config.getboolean('data', 'oversample')
    #n_sample = int(config['data']['n_sample'])
    algo = config['train']['algo']

    if algo == 'knn' or algo == 'rf' or algo == 'ha' or algo == 'lr' or algo == 'svm':
        baseline_approach = Baseline()

    horizons = config.get("data", "input_horizons")
    ip_horizons = json.loads(horizons)

    for ip_horizon in ip_horizons:
        ip_horizon = int(ip_horizon)
        if feat:
            X_train, Y_train, Train_cs_features, Train_spatial_features, Train_pattern_features, \
            Train_median_features, Train_q25_features, Train_q75_features = \
                data_obj.split_train_test(df, ip_horizon)
        else:
            X_train, Y_train = data_obj.split_train_test(df, ip_horizon)
            Train_cs_features = np.random.rand(len(X_train), 2)
            Train_spatial_features = np.random.rand(len(X_train), 2)
            Train_pattern_features = np.random.rand(len(X_train), 2)
            Train_median_features = np.random.rand(len(X_train), 2)
            Train_q25_features = np.random.rand(len(X_train), 2)
            Train_q75_features = np.random.rand(len(X_train), 2)

        print(len(X_train[0]), len(Y_train[0]))

        if train_:

            if algo == 'knn':
                baseline_approach.nearest_neighbour(X_train, Y_train, ip_horizon)
            elif algo == 'rf':
                baseline_approach.random_forest(X_train, Y_train, ip_horizon)
            elif algo == 'lr':
                baseline_approach.logistic_regression_classifier(X_train, Y_train, ip_horizon)
            elif algo == 'svm':
                baseline_approach.svm_classifier(X_train, Y_train, ip_horizon)
            elif algo == 'ha':
                pass
            else:
                train(config, X_train, Y_train, Train_cs_features, Train_spatial_features, Train_pattern_features,
                      Train_median_features, Train_q25_features, Train_q75_features, ip_horizon)

        if eval_train:
            evaluate(config, X_train, Y_train, Train_cs_features, Train_spatial_features, Train_pattern_features, X_train.shape[0], ip_horizon)

        if eval_tests:
            if feat:
                X, Y, Feat_cs, Feat_spatial, Feat_pattern, Feat_median, Feat_q25, Feat_q75 = \
                    generate_test_set(config, ip_horizon)
            else:
                X, Y = generate_test_set(config, ip_horizon)
                Feat_cs = [np.random.rand(len(X[0]), 2)] * 5
                Feat_spatial = [np.random.rand(len(X[0]), 2)] * 5
                Feat_pattern = [np.random.rand(len(X[0]), 2)] * 5
                Feat_median = [np.random.rand(len(X[0]), 2)] * 5
                Feat_q25 = [np.random.rand(len(X[0]), 2)] * 5
                Feat_q75 = [np.random.rand(len(X[0]), 2)] * 5

            if algo == 'knn' or algo == 'rf' or algo == 'ha' or algo == 'lr' or algo == 'svm':
                baseline_approach.eval_test_set(len(X_train), X, Y, ip_horizon, df)
            else:
                evaluate_test_set(config, X, Y, Feat_cs, Feat_spatial, Feat_pattern, Feat_median, Feat_q25,
                                  Feat_q75, len(X_train), ip_horizon)














