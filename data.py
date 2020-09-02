import numpy as np
import pandas as pd
import os
import datetime
import sys
from configparser import ConfigParser
from sklearn.utils import shuffle

class Data:

    def __init__(self):
        self._config = ConfigParser()
        self._config.read('config.ini')

    '''
    Reads the data from path mentioned in config.
    :parameter
    
    :returns
    dataframe containing the data
    '''
    def load_data(self):
        data_dir = self._config['data']['path']

        df = pd.read_csv(os.path.join(data_dir, 'train_val.tsv'), sep='\t', header=None,
                         names=['identifier', 'outlet', 'usage_count', 'time_stamp'])

        df['time_stamp'] = pd.to_datetime(df['time_stamp'], infer_datetime_format=True)
        print('Min: ', df.time_stamp.min())
        print('Max: ', df.time_stamp.max())

        # switch rows to columns so that each column represents a series
        new_df = pd.pivot_table(df, values='usage_count',
                                index=['time_stamp'],
                                columns=['identifier', 'outlet'],
                                fill_value=0)

        # fill the missing values with 0
        new_df = new_df.reindex(pd.date_range(start=df.time_stamp.min(), end=df.time_stamp.max(), freq='15min'),
                                fill_value=0)

        new_df = new_df.loc[self._config['data']['train_start']:self._config['data']['val_stop']]

        return new_df

    # @refrence: https://machinelearningmastery.com/how-to-develop-machine-learning-models-for-multivariate-multi-step-air-pollution-time-series-forecasting/
    def split_train_test(self, series, n_lag=96, n_lead=96, train_step_size=96, test_step_size=96, randomize=True):
        '''
        :param n_lead: forecast length
        :param series: pandas series containing Time series data
        :param n_lag: Number of lags.
        :param train_step_size: steps to move in rolling window to generate new ip/op pattern in training data
        :param test_step_size: steps to move in rolling window to generate new ip/op pattern in test data
        :return:
            X_train - input patterns in train data
            Y_train - output pattern (label) for train data
            X_test - input patterns for val data
            Y_test - output pattern for validation data
        '''
        X_train = list()
        X_test = list()
        Y_train = list()
        Y_test = list()

        split_time = self._config['data']['train_stop']

        for i in range(n_lag, series.size, train_step_size):
            end_ix = i + (n_lead - 1)
            # check if can create a pattern
            if end_ix >= series.size:
                break
            # retrieve input and output
            start_ix = i - n_lag

            if series.index[end_ix] <= datetime.datetime.strptime(split_time, '%Y-%m-%d %H:%M:%S'):
                X_train.append(series.tolist()[start_ix:i])
                Y_train.append(series.tolist()[i:(end_ix + 1)])
            else:
                if i%test_step_size == 0:
                    X_test.append(series.tolist()[start_ix:i])
                    Y_test.append(series.tolist()[i:(end_ix + 1)])
        # shuffle
        if randomize:
            X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
            X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

        return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

    def generate_and_save_aggregated_train_test(self, df, n_lag_days=1, n_lead_days=1, train_step_size=1, test_step_size=1, randomize=True):

        '''
        Generates the train and test data by aggregating the data of all the series and save the data as npy file
        :param df: datafrme containging all the series
        :param lag: input horizon (number of days)
        :return:
        '''

        save_dir = self._config['data']['npy_path']

        X_train = list()
        Y_train = list()
        X_test = list()
        Y_test = list()

        for series_idx in range(df.shape[1]):
            x_train, y_train, x_test, y_test = self.split_train_test(series=df.iloc[:, series_idx],
                                                                         n_lag= 96*n_lag_days, n_lead=96*n_lead_days,
                                                                         train_step_size=train_step_size, test_step_size=test_step_size)
            X_train.extend(x_train.tolist())
            Y_train.extend(y_train.tolist())
            X_test.extend(x_test.tolist())
            Y_test.extend(y_test.tolist())

            print(series_idx, sep=' ', end=' ')
            sys.stdout.flush()

        # shuffle and save the data
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)

        if randomize:
            X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
            X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

        np.save(os.path.join(save_dir, 'X_train_lag_' + str(n_lag_days) + '_day.npy'), X_train)
        np.save(os.path.join(save_dir, 'Y_train_lag_' + str(n_lag_days) + '_day.npy'), Y_train)
        np.save(os.path.join(save_dir, 'X_test_lag_' + str(n_lag_days) + '_day.npy'), X_test)
        np.save(os.path.join(save_dir, 'Y_test_lag_' + str(n_lag_days) + '_day.npy'), Y_test)

        print('Finished Generating : ', str(n_lag_days))
        sys.stdout.flush()

    def load_npy(self, input_horizon):

        npy_path = self._config['data']['npy_path']
        X_train = np.load(os.path.join(npy_path, 'X_train_lag_' + str(input_horizon) + '_day.npy'))
        Y_train = np.load(os.path.join(npy_path, 'Y_train_lag_' + str(input_horizon) + '_day.npy'))
        X_test = np.load(os.path.join(npy_path, 'X_test_lag_' + str(input_horizon) + '_day.npy'))
        Y_test = np.load(os.path.join(npy_path, 'Y_test_lag_' + str(input_horizon) + '_day.npy'))

        return X_train, Y_train, X_test, Y_test



