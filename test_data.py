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
import multiprocessing
from functools import partial


def split_test_set(idx, config, data_obj, df, cs_feature, spatial_feature, pattern_feature, weekday_feature,
                   weekend_feature, median_feature, q25_feature, q75_feature, start_date, stop_date, n_lag):

    X_test = list()
    Y_test = list()
    test_features_cs = list()
    test_features_spatial = list()
    test_features_pattern = list()
    test_features_median = list()
    test_features_q25 = list()
    test_features_q75 = list()

    test_step = int(config['data']['test_window_size'])
    #n_lag = int(config['data']['input_horizon'])  # *96 when lag value is in days
    n_lead = int(config['data']['output_horizon'])  # *96 when lead is in days
    feat = config.getboolean('data', 'features')

    series = df.iloc[:, idx]
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
                data, features_cs, features_spatial, features_pattern, features_median, features_q25, features_q75 = \
                    data_obj.generate_data(series, df, cs_feature, spatial_feature, pattern_feature,
                                           weekday_feature, weekend_feature, median_feature, q25_feature, q75_feature,
                                           start_ix, i, i, (end_ix + 1))
                X_test.append(data.tolist())
                test_features_cs.append(features_cs.tolist())
                test_features_spatial.append(features_spatial.tolist())
                test_features_pattern.append(features_pattern)
                test_features_median.append(features_median)
                test_features_q25.append(features_q25)
                test_features_q75.append(features_q75)
            else:
                X_test.append(series.tolist()[start_ix:i])
            Y_test.append(series.tolist()[i:(end_ix + 1)])

    print(idx, sep=' ', end=' ')
    sys.stdout.flush()

    if feat:
        X_test, Y_test, test_features_cs, test_features_spatial, test_features_pattern, test_features_median,\
            test_features_q25, test_features_q75 = \
            shuffle(X_test, Y_test, test_features_cs, test_features_spatial, test_features_pattern,
                    test_features_median, test_features_q25, test_features_q75, random_state=0)
        return np.array(X_test), np.array(Y_test), np.array(test_features_cs), np.array(test_features_spatial),\
               np.array(test_features_pattern), np.array(test_features_median), \
               np.array(test_features_q25), np.array(test_features_q75)
    else:
        X_test, Y_test = shuffle(X_test, Y_test, random_state=0)
        return np.array(X_test), np.array(Y_test)


def generate_and_save(config, test_idx, data_obj,  df, cs_feature, spatial_feature, pattern_feature, weekday_feature,
                      weekend_feature, median_feature, quant_25_feature, quant_75_feature, input_horizon):
    #input_horizon = int(config['data']['input_horizon'])
    test_path = config['data']['test_path']
    feat = config.getboolean('data', 'features')
    n_core = int(config['data']['n_core'])

    X = list()
    Y = list()
    feat_cs = list()
    feat_spatial = list()
    feat_pattern = list()
    feat_median = list()
    feat_q25 = list()
    feat_q75 = list()
    start_date = config['test']['test' + str(test_idx+1) + '_start']
    stop_date = config['test']['test' + str(test_idx+1) + '_stop']

    #series_param = range(5)
    series_param = range(df.shape[1] - 35)
    pool = multiprocessing.Pool(processes=n_core)
    multi_func = partial(split_test_set, config=config, data_obj=data_obj, df=df, cs_feature=cs_feature, spatial_feature=spatial_feature,
                         pattern_feature=pattern_feature, weekday_feature=weekday_feature,
                         weekend_feature=weekend_feature,
                         median_feature=median_feature, q25_feature=quant_25_feature,
                         q75_feature=quant_75_feature, start_date=start_date, stop_date=stop_date,
                         n_lag=input_horizon)
    result_list = pool.map(multi_func, series_param)
    pool.close()
    pool.join()
    print(len(result_list[0]), len(result_list[0][0]), len(result_list[0][2][0]))

    for i in range(len(result_list)):
        x = result_list[i][0]
        y = result_list[i][1]
        if feat:
            feat_cs.extend(result_list[i][2])
            feat_spatial.extend(result_list[i][3])
            feat_pattern.extend(result_list[i][4])
            feat_median.extend(result_list[i][5])
            feat_q25.extend(result_list[i][6])
            feat_q75.extend(result_list[i][7])
        X.extend(x)
        Y.extend(y)

    print('Finished test set ', str(test_idx+1))
    if feat:
        return X, Y, feat_cs, feat_spatial, feat_pattern, feat_median, feat_q25, feat_q75
    else:
        return X, Y


def generate_new_data(df, start, mid, stop):

    x = df.iloc[:, :-35][start:mid].to_numpy()
    y = df.iloc[:, :-35][mid:stop + 1].to_numpy()
    return x, y


def split_df_test(config, df, start_date, stop_date):

    X_test = list()
    Y_test = list()

    test_step = int(config['data']['test_window_size'])
    n_lag = int(config['data']['input_horizon'])
    n_lead = int(config['data']['output_horizon'])

    for i in range(n_lag, len(df)):
        end_ix = i + (n_lead - 1)
        # check if can create a pattern
        if end_ix >= len(df):
            break
        # retrieve input and output
        start_ix = i - n_lag

        if (i % test_step == 0) and (df.index[end_ix] > datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')) and \
                (df.index[end_ix] <= datetime.datetime.strptime(stop_date, '%Y-%m-%d %H:%M:%S')):
            X, Y = generate_new_data(df, start_ix, i, end_ix)
            X_test.append(X)
            Y_test.append(Y)

    return np.array(X_test), np.array(Y_test)


def generate_and_save_new(config, folder_list):

    input_horizon = int(config['data']['input_horizon'])
    test_path = config['data']['test_path']
    start = config['test']['start']
    stop = config['test']['test5_stop']

    data_obj = Data()
    df = data_obj.read_tsv('aug_dec_no_filter.tsv', start, stop)

    for i, val in enumerate(folder_list):

        start_date = config['test']['test' + str(i + 1) + '_start']
        stop_date = config['test']['test' + str(i + 1) + '_stop']

        X, Y = split_df_test(config, df, start_date, stop_date)

        np.save(os.path.join(test_path, folder_list[i], 'X_lag_' + str(input_horizon) + '.npy'), np.array(X))
        np.save(os.path.join(test_path, folder_list[i], 'Y_lag_' + str(input_horizon) + '.npy'), np.array(Y))

        print('Finished test set ', str(i + 1))


def generate_test_set(config, input_horizon):

    #input_horizon = int(config['data']['input_horizon'])
    test_path = config['data']['test_path']
    feat = config.getboolean('data', 'features')
    start = config['test']['start']
    stop = config['test']['test5_stop']

    data_obj = Data()
    df = data_obj.read_tsv('aug_dec_no_filter.tsv', start, stop)
    cs_feature, spatial_feature = data_obj.read_and_process_features()
    pattern_feature, weekday_feature, weekend_feature, median_feature, \
    quant_25_feature, quant_75_feature = data_obj.gen_pattern_features(df=df)

    X = list()
    Y = list()
    Feat_cs = list()
    Feat_spatial = list()
    Feat_pattern = list()
    Feat_median = list()
    Feat_q25 = list()
    Feat_q75 = list()

    if feat:
        folder_list = ['test1_data_time', 'test2_data_time', 'test3_data_time', 'test4_data_time', 'test5_data_time']
    else:
        folder_list = ['test1_data', 'test2_data', 'test3_data', 'test4_data', 'test5_data']

    for i, val in enumerate(folder_list):
        if feat:
            X_, Y_, cs, spatial, pattern, median, q25, q75 = generate_and_save(config, i, data_obj,  df, cs_feature,
                                                                               spatial_feature, pattern_feature,
                                                                               weekday_feature,
                                                                               weekend_feature, median_feature,
                                                                               quant_25_feature, quant_75_feature,
                                                                               input_horizon)
            X.append(X_)
            Y.append(Y_)
            Feat_cs.append(cs)
            Feat_spatial.append(spatial)
            Feat_pattern.append(pattern)
            Feat_median.append(median)
            Feat_q25.append(q25)
            Feat_q75.append(q75)
        else:
            X_, Y_= generate_and_save(config, i, data_obj, df, cs_feature, spatial_feature, pattern_feature,
                                      weekday_feature, weekend_feature, median_feature,
                                      quant_25_feature, quant_75_feature, input_horizon)
            X.append(X_)
            Y.append(Y_)

    print('Finished Lag:', str(input_horizon))
    if feat:
        return X, Y, Feat_cs, Feat_spatial, Feat_pattern, Feat_median, Feat_q25, Feat_q75
    else:
        return X, Y


def evaluate_test_set(config, X, Y, Feat_cs, Feat_spatial, Feat_pattern, Feat_median, Feat_q25, Feat_q75, n_train, input_horizon):

    prec_0 = list()
    prec_1 = list()
    rec_0 = list()
    rec_1 = list()
    f1_0 = list()
    f1_1 = list()

    result_path = config['result']['path']
    #input_horizon = int(config['data']['input_horizon'])
    comment = config['result']['comment']
    output_horizon = int(config['data']['output_horizon'])

    for i in range(len(X)):
        pred, target = evaluate(config, X[i], Y[i], Feat_cs[i], Feat_spatial[i], Feat_pattern[i], Feat_median[i],
                                Feat_q25[i], Feat_q75[i], n_train, input_horizon)
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

        result_row = [n_train, len(X[i]), input_horizon, output_horizon, th[ix],
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

    result_row = [n_train, len(X[i]), input_horizon, output_horizon, np.mean(prec_0), np.mean(prec_1),
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





