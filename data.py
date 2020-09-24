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
    def read_tsv(self):
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
    def split_series_train_test(self, series, randomize=True):
        '''
        :param series: pandas series containing Time series data
        :param randomize: if true shuffle the data
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
        train_step = int(self._config['data']['train_window_size'])
        test_step = int(self._config['data']['test_window_size'])
        n_lag = 96 * int(self._config['data']['input_horizon'])
        n_lead = 96 * int(self._config['data']['output_horizon'])

        for i in range(n_lag, series.size):
            end_ix = i + (n_lead - 1)
            # check if can create a pattern
            if end_ix >= series.size:
                break
            # retrieve input and output
            start_ix = i - n_lag

            if (i%train_step == 0) and (series.index[end_ix] <= datetime.datetime.strptime(split_time, '%Y-%m-%d %H:%M:%S')):
                X_train.append(series.tolist()[start_ix:i])
                Y_train.append(series.tolist()[i:(end_ix + 1)])
            elif (i%test_step == 0):
                X_test.append(series.tolist()[start_ix:i])
                Y_test.append(series.tolist()[i:(end_ix + 1)])

        # shuffle
        if randomize:
            X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
            X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

        return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

    def generate_and_save_aggregated_train_test(self, df, randomize=True):

        '''
        Generates the train and test data by aggregating the data of all the series and save the data as npy file
        :param df: datafrme containging all the series
        :param randomize: if true shuffle the data
        :return:
        '''

        save_dir = self._config['data']['npy_path']
        n_lag_days = int(self._config['data']['input_horizon'])
        n_lead_days = int(self._config['data']['output_horizon'])

        X_train = list()
        Y_train = list()
        X_test = list()
        Y_test = list()

        for series_idx in range(df.shape[1]):
            x_train, y_train, x_test, y_test = self.split_series_train_test(series=df.iloc[:, series_idx],
                                                                     randomize=randomize)
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

        train_step = self._config['data']['train_window_size']
        test_step = self._config['data']['test_window_size']

        np.save(os.path.join(save_dir, 'X_train_lag_' + str(n_lag_days) +
                             '_day_lead_' + str(n_lead_days) +
                             '_day_train step_' + str(train_step) +
                             '_day_test step_' + str(test_step) + '.npy'), X_train)

        np.save(os.path.join(save_dir, 'Y_train_lag_' + str(n_lag_days) +
                             '_day_lead_' + str(n_lead_days) +
                             '_day_train_step_' + str(train_step) +
                             '_day_test_step_' + str(test_step) + '.npy'), Y_train)

        np.save(os.path.join(save_dir, 'X_test_lag_' + str(n_lag_days) +
                             '_day_lead_' + str(n_lead_days) +
                             '_day_train_step_' + str(train_step) +
                             '_day_test_step_' + str(test_step) + '.npy'), X_test)

        np.save(os.path.join(save_dir, 'Y_test_lag_' + str(n_lag_days) +
                             '_day_lead_' + str(n_lead_days) +
                             '_day_train_step_' + str(train_step) +
                             '_day_test_step_' + str(test_step) + '.npy'), Y_test)

        print('Finished Generating : ', str(n_lag_days))
        sys.stdout.flush()

    def split_train_test(self, df, series_idx, aggregate=True, randomize=True):

        '''
        :param df: Dataframe containing all time series
        :param series_idx: index of series for which
        :param aggregate: if true function will aggregate the series and then return the train/test split
        :param randomize: if True shuffles the data
        :return: Train/Test split of data as numpy array
        '''

        input_horizon = int(self._config['data']['input_horizon'])
        output_horizon = int(self._config['data']['output_horizon'])

        if not aggregate:
            return self.split_series_train_test(df.iloc[:, series_idx], randomize=randomize)

        train_step = self._config['data']['train_window_size']
        test_step = self._config['data']['test_window_size']
        npy_path = self._config['data']['npy_path']

        if not os.path.isfile(os.path.join(npy_path, 'X_train_lag_' + str(input_horizon) +
                                                     '_day_lead_' + str(output_horizon) +
                                                     '_day_train step_' + str(train_step) +
                                                     '_day_test step_' + str(test_step) + '.npy')):
            self.generate_and_save_aggregated_train_test(df=df, randomize=randomize)

        X_train = np.load(os.path.join(npy_path, 'X_train_lag_' + str(input_horizon) +
                             '_day_lead_' + str(output_horizon) +
                             '_day_train step_' + str(train_step) +
                             '_day_test step_' + str(test_step) + '.npy'))

        Y_train = np.load(os.path.join(npy_path, 'Y_train_lag_' + str(input_horizon) +
                             '_day_lead_' + str(output_horizon) +
                             '_day_train_step_' + str(train_step) +
                             '_day_test_step_' + str(test_step) + '.npy'))

        X_test = np.load(os.path.join(npy_path, 'X_test_lag_' + str(input_horizon) +
                             '_day_lead_' + str(output_horizon) +
                             '_day_train_step_' + str(train_step) +
                             '_day_test_step_' + str(test_step) + '.npy'))

        Y_test = np.load(os.path.join(npy_path, 'Y_test_lag_' + str(input_horizon) +
                             '_day_lead_' + str(output_horizon) +
                             '_day_train_step_' + str(train_step) +
                             '_day_test_step_' + str(test_step) + '.npy'))

        return X_train, Y_train, X_test, Y_test



