import numpy as np
import pandas as pd
import os
import datetime
from data import Data
import sys
from train import evaluate
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils import shuffle

def split_test_set(config, data_obj, series, df, feature_df,  start_date, stop_date):

    X_test = list()
    Y_test = list()
    test_features = list()

    test_step = int(config['data']['test_window_size'])
    n_lag = 96 * int(config['data']['input_horizon'])  # *96 when lag value is in days
    n_lead = 96 * int(config['data']['output_horizon'])  # *96 when lead is in days
    feat = config.getboolean('data', 'features')

    for i in range(n_lag, series.size):
        end_ix = i + (n_lead - 1)
        # check if can create a pattern
        if end_ix >= series.size:
            break
        # retrieve input and output
        start_ix = i - n_lag

        if (i % test_step == 0) and (series.index[end_ix] > datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')) and \
                (series.index[end_ix] <= datetime.datetime.strptime(stop_date, '%Y-%m-%d %H:%M:%S')):

            if feat:
                data, features = data_obj.generate_data(series, df, feature_df, start_ix, i)
                X_test.append(data.tolist())
                test_features.append(features.tolist())
            else:
                X_test.append(series.tolist()[start_ix:i])
            Y_test.append(series.tolist()[i:(end_ix + 1)])
            # data, _ = self.generate_data(series, df, feature_df, i, (end_ix + 1))
            # Y_train.append(data.tolist())

    if feat:
        X_test, Y_test, test_features = shuffle(X_test, Y_test, test_features, random_state=0)
        return np.array(X_test), np.array(Y_test), np.array(test_features)
    else:
        X_test, Y_test = shuffle(X_test, Y_test, random_state=0)
        return np.array(X_test), np.array(Y_test)


def generate_and_save(config, folder_list):
    input_horizon = int(config['data']['input_horizon'])
    test_path = config['data']['test_path']
    feat = config.getboolean('data', 'features')
    start = config['test']['start']
    stop = config['test']['test5_stop']

    data_obj = Data()
    df = data_obj.read_tsv('test.tsv', start, stop)
    feature_df = data_obj.read_and_process_features()

    for i, val in enumerate(folder_list):
        X = list()
        Y = list()
        feature = list()
        start_date = config['test']['test' + str(i+1) + '_start']
        stop_date = config['test']['test' + str(i+1) + '_stop']

        for series_idx in range(df.shape[1] - 9):
            if feat:
                x, y, f = split_test_set(config, data_obj, df.iloc[:, series_idx], df, feature_df, start_date, stop_date)
                feature.extend(f)
            else:
                x, y = split_test_set(config, data_obj, df.iloc[:, series_idx], df, feature_df, start_date, stop_date)

            X.extend(x)
            Y.extend(y)

            print(series_idx, sep=' ', end=' ')
            sys.stdout.flush()

        np.save(os.path.join(test_path, folder_list[i], 'X_lag_' + str(input_horizon) + '.npy'), np.array(X))
        np.save(os.path.join(test_path, folder_list[i], 'Y_lag_' + str(input_horizon) + '.npy'), np.array(Y))

        if feat:
            np.save(os.path.join(test_path, folder_list[i], 'Feature_lag_' + str(input_horizon) + '.npy'), np.array(feature))

        print('Finished test set ', str(i+1))

    print('Finished Lag:', str(input_horizon))



def generate_test_set(config):

    input_horizon = int(config['data']['input_horizon'])
    test_path = config['data']['test_path']
    feat = config.getboolean('data', 'features')

    X = list()
    Y = list()
    Feat = list()

    if feat:
        folder_list = ['test1_data_time', 'test2_data_time', 'test3_data_time', 'test4_data_time', 'test5_data_time']
    else:
        folder_list = ['test1_data', 'test2_data', 'test3_data', 'test4_data', 'test5_data']

    if not os.path.isfile(os.path.join(test_path, folder_list[0], 'X_lag_' + str(input_horizon) + '.npy')):
        generate_and_save(config, folder_list)

    # loop thorugh folder, load npy and appent to X list
    for i, val in enumerate(folder_list):
        X.append(np.load(os.path.join(test_path, folder_list[i], 'X_lag_' + str(input_horizon) + '.npy')))
        Y.append(np.load(os.path.join(test_path, folder_list[i], 'Y_lag_' + str(input_horizon) + '.npy')))

        if feat:
            Feat.append(np.load(os.path.join(test_path, folder_list[i], 'Feature_lag_' + str(input_horizon) + '.npy')))
    if feat:
        return X, Y, Feat
    else:
        return X, Y

def evaluate_test_set(config, X, Y, Feat, n_train):

    for i in range(len(X)):
        pred, target = evaluate(config, X[i], Y[i], Feat[i], n_train)
        prec, rec, th = precision_recall_curve(target.ravel(), pred.ravel())
        print('threshold: ')
        print(th)
        fscore = (2 * prec * rec) / (prec + rec)
        fscore = np.nan_to_num(fscore)
        ix = np.argmax(fscore)
        print('Best Threshold=%f, F-Score=%.3f' % (th[ix], fscore[ix]))