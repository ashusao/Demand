import numpy as np
import pandas as pd
import os
import datetime
from data import Data
import sys
import csv
from train import evaluate
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import shuffle
from dd_data import read_and_convert_dd
from dd_data import gen_pattern_features_
from dd_data import generate_data_


def split_test_set_(config, data_obj, series, df, pattern_feature, start_date, stop_date):

    X_test = list()
    Y_test = list()
    test_features_pattern = list()

    test_step = int(config['data']['test_window_size'])
    n_lag = int(config['data']['input_horizon'])  # *96 when lag value is in days
    n_lead = int(config['data']['output_horizon'])  # *96 when lead is in days
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
                data, features_pattern = generate_data_(series, df, pattern_feature, start_ix, i)
                X_test.append(data.tolist())
                test_features_pattern.append(features_pattern.tolist())
            else:
                X_test.append(series.tolist()[start_ix:i])
            Y_test.append(series.tolist()[i:(end_ix + 1)])
            # data, _ = self.generate_data(series, df, feature_df, i, (end_ix + 1))
            # Y_train.append(data.tolist())

    if feat:
        X_test, Y_test, test_features_pattern = shuffle(X_test, Y_test, test_features_pattern, random_state=0)
        return np.array(X_test), np.array(Y_test), np.array(test_features_pattern)
    else:
        X_test, Y_test = shuffle(X_test, Y_test, random_state=0)
        return np.array(X_test), np.array(Y_test)


def generate_and_save_(config, folder_list):
    input_horizon = int(config['data']['input_horizon'])
    test_path = config['data']['test_path']
    feat = config.getboolean('data', 'features')
    start = config['test']['start']
    stop = config['test']['test4_stop']
    train_start = config['data']['train_start']
    train_stop = config['data']['train_stop']

    data_obj = Data()
    #df = data_obj.read_tsv('aug_dec_no_filter.tsv', start, stop)
    df = read_and_convert_dd(config, 'dataset.xlsx', start, stop)
    pattern_feature = gen_pattern_features_(config, df=read_and_convert_dd(config, 'dataset.xlsx', train_start, train_stop))

    for i, val in enumerate(folder_list):
        X = list()
        Y = list()
        feat_pattern = list()
        start_date = config['test']['test' + str(i+1) + '_start']
        stop_date = config['test']['test' + str(i+1) + '_stop']

        for series_idx in range(df.shape[1] - 35):
            if feat:
                x, y, f_pattern = split_test_set_(config, data_obj, df.iloc[:, series_idx], df,
                                                                   pattern_feature, start_date, stop_date)
                feat_pattern.extend(f_pattern)
            else:
                x, y = split_test_set_(config, data_obj, df.iloc[:, series_idx], df,
                                       pattern_feature, start_date, stop_date)

            X.extend(x)
            Y.extend(y)

            print(series_idx, sep=' ', end=' ')
            sys.stdout.flush()

        np.save(os.path.join(test_path, folder_list[i], 'X_lag_' + str(input_horizon) + '.npy'), np.array(X))
        np.save(os.path.join(test_path, folder_list[i], 'Y_lag_' + str(input_horizon) + '.npy'), np.array(Y))

        if feat:
            np.save(os.path.join(test_path, folder_list[i], 'Pattern_lag_' + str(input_horizon) + '.npy'), np.array(feat_pattern))

        print('Finished test set ', str(i+1))

    print('Finished Lag:', str(input_horizon))


def generate_test_set_(config):

    input_horizon = int(config['data']['input_horizon'])
    test_path = config['data']['test_path']
    feat = config.getboolean('data', 'features')

    X = list()
    Y = list()
    Feat_pattern = list()

    if feat:
        folder_list = ['test1_data_time', 'test2_data_time', 'test3_data_time', 'test4_data_time']
    else:
        folder_list = ['test1_data', 'test2_data', 'test3_data', 'test4_data']

    if not os.path.isfile(os.path.join(test_path, folder_list[0], 'X_lag_' + str(input_horizon) + '.npy')):
        generate_and_save_(config, folder_list)

    # loop thorugh folder, load npy and appent to X list
    for i, val in enumerate(folder_list):
        X.append(np.load(os.path.join(test_path, folder_list[i], 'X_lag_' + str(input_horizon) + '.npy')))
        Y.append(np.load(os.path.join(test_path, folder_list[i], 'Y_lag_' + str(input_horizon) + '.npy')))

        if feat:
            Feat_pattern.append(np.load(os.path.join(test_path, folder_list[i], 'Pattern_lag_' + str(input_horizon) + '.npy')))
    if feat:
        return X, Y, Feat_pattern
    else:
        return X, Y

def evaluate_test_set_(config, X, Y, Feat_cs, Feat_spatial, Feat_pattern, n_train):

    prec_0 = list()
    prec_1 = list()
    rec_0 = list()
    rec_1 = list()
    f1_0 = list()
    f1_1 = list()

    result_path = config['result']['path']
    input_horizon = int(config['data']['input_horizon'])
    comment = config['result']['comment']
    output_horizon = int(config['data']['output_horizon'])


    for i in range(len(X)):
        pred, target = evaluate(config, X[i], Y[i], Feat_cs[i], Feat_spatial[i], Feat_pattern[i], n_train)
        prec, rec, th = precision_recall_curve(target.ravel(), pred.ravel())
        print('threshold: ')
        print(th)
        fscore = (2 * prec * rec) / (prec + rec)
        fscore = np.nan_to_num(fscore)
        ix = np.argmax(fscore)
        print('Best Threshold=%f, F-Score=%.3f' % (th[ix], fscore[ix]))

        prediction = np.copy(pred)
        prediction[prediction >= th[ix]] = 1
        prediction[prediction < th[ix]] = 0

        precision, recall, f1, _ = precision_recall_fscore_support(target.ravel(), prediction.ravel(), average=None)
        prec_0.append(precision[0])
        prec_1.append(precision[1])
        rec_0.append(recall[0])
        rec_1.append(recall[1])
        f1_0.append(f1[0])
        f1_1.append(f1[1])

        result_row = [n_train, X[i].shape[0], input_horizon, output_horizon, th[ix],
                      precision[0], precision[1], recall[0], recall[1], f1[0], f1[1], comment]

        result_file = os.path.join(result_path, 'test_set_' + str(i+1) + '.csv')

        if not os.path.isfile(result_file):
            header = ['n_train', 'n_test', 'input_horizon', 'output_horizon', 'threshold',
                      'prec_0', 'prec_1', 'rec_0', 'rec_1', 'F1_0', 'F1_1', 'comment']

            with open(result_file, "a+", newline='') as f:
                csv_writer = csv.writer(f, delimiter=',')
                csv_writer.writerow(header)
                csv_writer.writerow(result_row)
        else:
            with open(result_file, "a+", newline='') as f:
                csv_writer = csv.writer(f, delimiter=',')
                csv_writer.writerow(result_row)

    result_file = os.path.join(result_path, 'avg_test_set.csv')

    prec_0 = np.array(prec_0)
    prec_1 = np.array(prec_1)
    rec_0 = np.array(rec_0)
    rec_1 = np.array(rec_1)
    f1_0 = np.array(f1_0)
    f1_1 = np.array(f1_1)

    result_row = [n_train, X[i].shape[0], input_horizon, output_horizon, np.mean(prec_0), np.mean(prec_1),
                  np.mean(rec_0), np.mean(rec_1), np.mean(f1_0), np.mean(f1_1), comment]

    if not os.path.isfile(result_file):
        header = ['n_train', 'n_test', 'input_horizon', 'output_horizon',
                  'prec_0', 'prec_1', 'rec_0', 'rec_1', 'F1_0', 'F1_1', 'comment']

        with open(result_file, "a+", newline='') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(header)
            csv_writer.writerow(result_row)
    else:
        with open(result_file, "a+", newline='') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(result_row)





