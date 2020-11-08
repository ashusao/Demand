import numpy as np
import pandas as pd
import os
import datetime
import sys
from configparser import ConfigParser
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

class Data:

    def __init__(self):
        self._config = ConfigParser()
        self._config.read('config.ini')
        self.cat = [[0],[1]]
        self.enc = OneHotEncoder()
        self.enc.fit(self.cat)

    def encode_time(self, data, col):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col] / data[col].max())
        data[col + '_cos'] = np.cos(2 * np.pi * data[col] / data[col].max())
        return data

    '''
    Reads the data from path mentioned in config.
      
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

        # adding time features
        new_df['day'] = new_df.index.dayofweek
        new_df['hour'] = new_df.index.hour
        new_df['minute'] = new_df.index.minute
        new_df = self.encode_time(new_df, 'day')
        new_df = self.encode_time(new_df, 'hour')
        new_df = self.encode_time(new_df, 'minute')

        min_max_scalar = preprocessing.MinMaxScaler()
        new_df[['day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']] = min_max_scalar.fit_transform(new_df[['day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']])

        return new_df

    def generate_data(self, series, df, start, stop):
        d = np.expand_dims(series.to_numpy()[start:stop], axis=1)
        day_sin = np.expand_dims(df['day_sin'].to_numpy()[start:stop], axis=1)
        day_cos = np.expand_dims(df['day_cos'].to_numpy()[start:stop], axis=1)
        hour_sin = np.expand_dims(df['hour_sin'].to_numpy()[start:stop], axis=1)
        hour_cos = np.expand_dims(df['hour_cos'].to_numpy()[start:stop], axis=1)
        minute_sin = np.expand_dims(df['minute_sin'].to_numpy()[start:stop], axis=1)
        minute_cos = np.expand_dims(df['minute_cos'].to_numpy()[start:stop], axis=1)
        data = np.concatenate([d, day_sin, day_cos, hour_sin, hour_cos, minute_sin, minute_cos], axis=1)
        return data

    # @refrence: https://machinelearningmastery.com/how-to-develop-machine-learning-models-for-multivariate-multi-step-air-pollution-time-series-forecasting/
    def split_series_train_test(self, series, df, randomize=True):
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
                X_train.append(self.generate_data(series, df, start_ix, i).tolist())
                # X_train.append(series.tolist()[start_ix:i])
                Y_train.append(series.tolist()[i:(end_ix + 1)])
            elif (i%test_step == 0):
                X_test.append(self.generate_data(series, df, start_ix, i).tolist())
                Y_test.append(series.tolist()[i:(end_ix + 1)])

        # shuffle
        if randomize:
            X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
            X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

        return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

    def generate_and_save_aggregated_train_test(self, df, randomize=True):

        '''
        Generates the train and test data by aggregating the data of all the series and save the data as npy file
        :param df: datafrme containing all the series
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

        for series_idx in range(df.shape[1] - 9): # subtract last 9 time feature columns
            x_train, y_train, x_test, y_test = self.split_series_train_test(df.iloc[:, series_idx],
                                                                            df, randomize=randomize)
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
        :param series_idx: index of series to split
        :param aggregate: if true function will aggregate the series and then return the train/test split
        :param randomize: if True shuffles the data
        :return: Train/Test split of data as numpy array
        '''

        input_horizon = int(self._config['data']['input_horizon'])
        output_horizon = int(self._config['data']['output_horizon'])

        if not aggregate:
            return self.split_series_train_test(df.iloc[:, series_idx], df, randomize=randomize)

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

    def load_one_hot_train(self, X_train, Y_train, X_test, Y_test, train=True):
        '''

        :param X_train: shape (batch_size, input_horizon)
        :param Y_train: shape (batch_size, output_horizon)
        :return:
            X_train_one_hot: shape (batch_size, input_horizon, one_hot_size)
            Y_train_one_hot: shape (batch_size, output_horizon, one_hot_size)
        '''
        input_horizon = int(self._config['data']['input_horizon'])
        output_horizon = int(self._config['data']['output_horizon'])
        train_step = self._config['data']['train_window_size']
        test_step = self._config['data']['test_window_size']
        one_hot_path = self._config['data']['npy_one_hot']

        X_train_one_hot_path = 'X_train_lag_' + str(input_horizon) + '_day_lead_' + str(output_horizon) + '_day_train step_' + str(train_step) + '_day_test step_' + str(test_step) + '.npy'
        Y_train_one_hot_path = 'Y_train_lag_' + str(input_horizon) + '_day_lead_' + str(output_horizon) + '_day_train step_' + str(train_step) + '_day_test step_' + str(test_step) + '.npy'
        X_test_one_hot_path = 'X_test_lag_' + str(input_horizon) + '_day_lead_' + str(output_horizon) + '_day_train step_' + str(train_step) + '_day_test step_' + str(test_step) + '.npy'
        Y_test_one_hot_path = 'Y_test_lag_' + str(input_horizon) + '_day_lead_' + str(output_horizon) + '_day_train step_' + str(train_step) + '_day_test step_' + str(test_step) + '.npy'

        if train:
            if not os.path.isfile(os.path.join(one_hot_path, X_train_one_hot_path)):
                np.save(os.path.join(one_hot_path, X_train_one_hot_path), self.one_hot_transform(X_train))
            X_train_one_hot = np.load(os.path.join(one_hot_path, X_train_one_hot_path))

            if not os.path.isfile(os.path.join(one_hot_path, Y_train_one_hot_path)):
                np.save(os.path.join(one_hot_path, Y_train_one_hot_path), self.one_hot_transform(Y_train))
            Y_train_one_hot = np.load(os.path.join(one_hot_path, Y_train_one_hot_path))

            return X_train_one_hot, Y_train_one_hot

        else:
            if not os.path.isfile(os.path.join(one_hot_path, X_test_one_hot_path)):
                np.save(os.path.join(one_hot_path, X_test_one_hot_path), self.one_hot_transform(X_test))
            X_test_one_hot = np.load(os.path.join(one_hot_path, X_test_one_hot_path))

            if not os.path.isfile(os.path.join(one_hot_path, Y_test_one_hot_path)):
                np.save(os.path.join(one_hot_path, Y_test_one_hot_path), self.one_hot_transform(Y_test))
            Y_test_one_hot = np.load(os.path.join(one_hot_path, Y_test_one_hot_path))

            return X_test_one_hot, Y_test_one_hot


    def one_hot_transform(self, batch):
        '''
        Function takes an input sequence and transforms it to one hot encoded sequence data

        :param batch: 2d - input sequence of shape(batch_size,seq_len)
        :return: 3d - one hot encoded output of shape(batch_size, seq_len, one_hot_size)
        '''
        enc_res = list()
        for seq in batch:
            for x in seq:
                # transform each input in 2d since sklearn transform excepts input of 2d shape
                tmp = self.enc.transform(x.reshape(1,-1)).toarray()
                # flatten the transformed data
                tmp = tmp.flatten()
                enc_res.append(tmp)

        return np.array(enc_res).reshape(batch.shape[0], batch.shape[1], -1)

    def one_hot_inv_transform(self, batch):
        '''
        Function takes an one hot encoded sequence and transforms it to decoded data

        :param batch: 3d - one hot encoded output of shape(batch_size, seq_len, one_hot_size)
        :return: 2d - decoded sequence of shape(batch_size,seq_len)
        '''
        dec_res = list()
        for seq in batch:
            dec_res.append(self.enc.inverse_transform(seq))

        return np.array(dec_res).reshape(batch.shape[0], -1)



