import numpy as np
import pandas as pd
import os
import datetime
import sys
from configparser import ConfigParser
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import multiprocessing
from functools import partial
import time

class Data:

    def __init__(self):
        self._config = ConfigParser()
        self._config.read('config.ini')

    def encode_time(self, data, col):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col])
        data[col + '_cos'] = np.cos(2 * np.pi * data[col])
        return data

    '''
    Reads the data from path mentioned in config.
      
    :returns
    dataframe containing the data
    '''
    def read_tsv(self, f_name, start, stop):
        data_dir = self._config['data']['path']

        df = pd.read_csv(os.path.join(data_dir, f_name), sep='\t', header=None,
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

        new_df = new_df.loc[start:stop]

        # adding time features

        new_df['weekday'] = new_df.index.dayofweek
        new_df['hour'] = new_df.index.hour
        new_df['minute'] = new_df.index.minute
        #new_df = self.encode_time(new_df, 'weekday')
        #new_df = self.encode_time(new_df, 'hour')
        #new_df = self.encode_time(new_df, 'minute')
        new_df = pd.get_dummies(new_df,
                                prefix=['weekday', 'hour', 'minute'],
                                columns=['weekday', 'hour', 'minute'])

        return new_df

    def read_and_process_features(self):

        data_dir = self._config['data']['path']

        station_df = pd.read_csv(os.path.join(data_dir, 'station_info.csv'), header=None,
                                 names=['identifier', 'lat', 'lon', 'address', 'anschlusse', 'anschluss',
                                        'type', 'power', 'current', 'status', 'suitable_for', 'provider',
                                        'zugang', 'opening_hours', 'cost', 'payment', 'electricity', 'geom',
                                        'park_area', 'restaurant', 'cafe', 'fast_food', 'toilet', 'pub',
                                        'airport', 'railway', 'beach', 'sea', 'river',
                                        'residential', 'commercial', 'retail', 'industrial',
                                        'motorway', 'trunk', 'primary', 'secondary',
                                        'motorway_link', 'trunk_link', 'primary_link', 'secondary_link'])
        station_df.drop(['geom', 'address', 'status'], axis=1, inplace=True)

        station_df['restaurant'].where(~(station_df.restaurant > 0), other=1, inplace=True)
        station_df['cafe'].where(~(station_df.cafe > 0), other=1, inplace=True)
        station_df['fast_food'].where(~(station_df.fast_food > 0), other=1, inplace=True)
        station_df['toilet'].where(~(station_df.toilet > 0), other=1, inplace=True)
        station_df['pub'].where(~(station_df.pub > 0), other=1, inplace=True)

        # Fill NAN
        station_df.provider.fillna('gewerblich', inplace=True)
        station_df.payment.fillna('undefiniert', inplace=True)

        id_anschluss = station_df[['identifier', 'anschluss']]
        cs_feature = station_df[['identifier', 'type', 'cost', 'payment', 'suitable_for', 'zugang', 'anschlusse',
                                'power', 'current']]
        spatial_feature = station_df[['identifier', 'anschluss', 'park_area', 'railway', 'airport',
                                      'sea', 'commercial', 'retail', 'industrial']]

        # one hot encode
        dum = pd.get_dummies(cs_feature,
                             prefix=['identifier', 'type', 'cost', 'payment', 'suitable_for', 'zugang'],
                             columns=['identifier', 'type', 'cost', 'payment', 'suitable_for', 'zugang'])

        cs_feature = pd.concat([id_anschluss, dum], axis=1)
        cs_feature.power = cs_feature['power'].map(lambda x: str(x)[:-1])
        cs_feature.current = cs_feature['current'].map(lambda x: str(x)[:-1])

        cs_feature.power = cs_feature.power.astype('int64')
        cs_feature.current = cs_feature.current.astype('int64')
        cs_feature.anschlusse = cs_feature.anschlusse.astype('int64')

        scaler = MinMaxScaler()
        cs_feature[['anschlusse', 'power', 'current']] = \
            scaler.fit_transform(cs_feature[['anschlusse', 'power', 'current']])

        spatial_feature[['park_area', 'railway', 'airport',
                         'sea', 'commercial', 'retail', 'industrial']] = \
            scaler.fit_transform(spatial_feature[['park_area', 'railway', 'airport',
                         'sea', 'commercial', 'retail', 'industrial']])

        spatial_feature['airport'] = 1 - spatial_feature['airport']
        spatial_feature['railway'] = 1 - spatial_feature['railway']
        spatial_feature['sea'] = 1 - spatial_feature['sea']

        return cs_feature, spatial_feature

    def scale_pattern_feat(self, daily_usage):
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(daily_usage)
        scaled_df = pd.DataFrame(scaled_features, index=daily_usage.index, columns=daily_usage.columns)
        return scaled_df

    def gen_pattern_features(self, df):
        train_df = df.loc[self._config['data']['train_start'] : self._config['data']['train_stop']].iloc[:, :-35]
        scaled_df = self.scale_pattern_feat(train_df.groupby([train_df.index.hour, train_df.index.minute]).sum())

        weekday_df = train_df.loc[(train_df.index.dayofweek >= 0) & (train_df.index.dayofweek <= 4)]
        weekend_df = train_df.loc[(train_df.index.dayofweek >= 5) & (train_df.index.dayofweek <= 6)]
        scaled_weekday = self.scale_pattern_feat(weekday_df.groupby([weekday_df.index.hour, weekday_df.index.minute]).sum())
        scaled_weekend = self.scale_pattern_feat(weekend_df.groupby([weekend_df.index.hour, weekend_df.index.minute]).sum())

        median_df = train_df.groupby([train_df.index.hour, train_df.index.minute]).median()
        quant_25_df = train_df.groupby([train_df.index.hour, train_df.index.minute]).quantile(0.25)
        quant_75_df = train_df.groupby([train_df.index.hour, train_df.index.minute]).quantile(0.75)

        return scaled_df, scaled_weekday, scaled_weekend, median_df, quant_25_df, quant_75_df

    def generate_features(self, series, cs_feature, spatial_feature):
        idx = series.name
        cs_feat = cs_feature[(cs_feature['identifier'] == idx[0]) &
                              (cs_feature['anschluss'] == idx[1])].iloc[0, 2:]
        spatial_feat = spatial_feature[(spatial_feature['identifier'] == idx[0]) &
                             (spatial_feature['anschluss'] == idx[1])].iloc[0, 2:]

        return cs_feat, spatial_feat

    def gen_pattern_slices(self, series, pattern_feature, start, stop):
        hour = series.index.hour[start:stop].tolist()
        minute = series.index.minute[start:stop].tolist()
        return pattern_feature.loc[list(zip(hour, minute))][series.name].tolist()

    def gen_weekday_weekend_slices(self, series, weekday_feature, weekend_feature, start, stop):
        pattern_feat = list()
        for i in range(start, stop):
            if (series.index[i].dayofweek >= 0) and (series.index[i].dayofweek <= 4):
                #print('*', sep=' ', end=' ')
                pattern_feat.append(weekday_feature[series.name].loc[(series.index[i].hour, series.index[i].minute)])
            else:
                #print('=', sep='', end='')
                pattern_feat.append(weekend_feature[series.name].loc[(series.index[i].hour, series.index[i].minute)])
        return pattern_feat


    def generate_data(self, series, df, cs_feature, spatial_feature, pattern_feature, weekday_feature, weekend_feature,
                      median_feature, quant_25_feature, quant_75_feature, start, stop, start_y, stop_y):

        d = np.expand_dims(series.to_numpy()[start:stop], axis=1)

        # genertate station and spatial features
        cs_feat, spatial_feat = self.generate_features(series, cs_feature, spatial_feature)

        # 1 day series pattern
        #pattern_feat = pattern_df[series.name]

        # pattern slices of y values
        pattern_feat = self.gen_pattern_slices(series, pattern_feature, start_y, stop_y)
        median_feat = self.gen_pattern_slices(series, median_feature, start_y, stop_y)
        quant_25_feat = self.gen_pattern_slices(series, quant_25_feature, start_y, stop_y)
        quant_75_feat = self.gen_pattern_slices(series, quant_75_feature, start_y, stop_y)
        #pattern_feat = self.gen_weekday_weekend_slices(series, weekday_feature, weekend_feature, start_y, stop_y)

        time_feat = df.iloc[:, -35:].to_numpy()[start:stop]
        data = np.concatenate([d, time_feat], axis=1)
        return data, cs_feat, spatial_feat, pattern_feat, median_feat, quant_25_feat, quant_75_feat

    # @refrence: https://machinelearningmastery.com/how-to-develop-machine-learning-models-for-multivariate-multi-step-air-pollution-time-series-forecasting/
    def split_series_train_test(self, idx, df, cs_feature, spatial_feature, pattern_feature,
                                weekday_feature, weekend_feature, median_feature, quant_25_feature,
                                quant_75_feature, n_lag, randomize=True):
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
        Y_train = list()
        train_cs_features = list()
        train_spatial_features = list()
        train_pattern_features = list()
        train_median_features = list()
        train_quant_25_features = list()
        train_quant_75_features = list()

        start_time = self._config['data']['train_start']
        split_time = self._config['data']['train_stop']
        train_step = int(self._config['data']['train_window_size'])
        n_lead = int(self._config['data']['output_horizon']) # *96 when lead is in days
        feat = self._config.getboolean('data', 'features')

        series = df.iloc[:, idx]

        for i in range(n_lag, series.size):
            end_ix = i + (n_lead - 1)
            # check if can create a pattern
            if end_ix >= series.size:
                break
            # retrieve input and output
            start_ix = i - n_lag

            if (i%train_step == 0) and (series.index[end_ix] <= datetime.datetime.strptime(split_time, '%Y-%m-%d %H:%M:%S')):

                if series.index[end_ix] > \
                        (datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(days=8)):

                    if feat:
                        data, cs_feat, spatial_feat, pattern_feat, median_feat, quant_25_feat, quant_75_feat = \
                            self.generate_data(series, df, cs_feature, spatial_feature, pattern_feature,
                                                           weekday_feature, weekend_feature, median_feature,
                                                           quant_25_feature, quant_75_feature, start_ix, i, i, (end_ix + 1))
                        X_train.append(data.tolist())
                        train_cs_features.append(cs_feat.tolist())
                        train_spatial_features.append(spatial_feat.tolist())
                        train_pattern_features.append(pattern_feat)
                        train_median_features.append(median_feat)
                        train_quant_25_features.append(quant_25_feat)
                        train_quant_75_features.append(quant_75_feat)
                    else:
                        X_train.append(series.tolist()[start_ix:i])
                    Y_train.append(series.tolist()[i:(end_ix + 1)])

        # shuffle
        if randomize:

            if feat:
                X_train, Y_train, train_cs_features, train_spatial_features, train_pattern_features, \
                train_median_features, train_quant_25_features, train_quant_75_features = \
                    shuffle(X_train, Y_train, train_cs_features, train_spatial_features, train_pattern_features,
                            train_median_features, train_quant_25_features, train_quant_75_features, random_state=0)
            else:
                X_train, Y_train = shuffle(X_train, Y_train, random_state=0)

        print(idx, sep=' ', end=' ')
        sys.stdout.flush()

        if feat:
            return X_train, Y_train, train_cs_features, train_spatial_features, train_pattern_features, \
                   train_median_features, train_quant_25_features, train_quant_75_features
        else:
            return X_train, Y_train


    def generate_and_save_aggregated_train_test(self, df, input_horizon, randomize=True):

        '''
        Generates the train and test data by aggregating the data of all the series and save the data as npy file
        :param df: datafrme containing all the series
        :param randomize: if true shuffle the data
        :return:
        '''

        n_lag_days = input_horizon
        feat = self._config.getboolean('data', 'features')
        n_core = int(self._config['data']['n_core'])

        X_train = list()
        Y_train = list()
        Train_cs_features = list()
        Train_spatial_features = list()
        Train_pattern_features = list()
        Train_median_features = list()
        Train_quant_25_features = list()
        Train_quant_75_features = list()

        cs_feature, spatial_feature = self.read_and_process_features()
        pattern_feature, weekday_feature, weekend_feature, median_feature, \
        quant_25_feature, quant_75_feature = self.gen_pattern_features(df)
        print(pattern_feature.shape, weekday_feature.shape, weekend_feature.shape)

        #series_param = range(10)
        series_param = range(df.shape[1] - 35)
        pool = multiprocessing.Pool(processes=n_core)
        multi_func = partial(self.split_series_train_test, df=df, cs_feature=cs_feature, spatial_feature=spatial_feature,
                             pattern_feature=pattern_feature, weekday_feature=weekday_feature, weekend_feature=weekend_feature,
                             median_feature=median_feature, quant_25_feature=quant_25_feature, quant_75_feature=quant_75_feature,
                             n_lag=n_lag_days, randomize=randomize)
        result_list = pool.map(multi_func, series_param)
        pool.close()
        pool.join()
        #print(len(result_list[0]), len(result_list[0][0]), len(result_list[0][2][0]))

        for i in range(len(result_list)):
            x_train = result_list[i][0]
            y_train = result_list[i][1]
            if feat:
                Train_cs_features.extend(result_list[i][2])
                Train_spatial_features.extend(result_list[i][3])
                Train_pattern_features.extend(result_list[i][4])
                Train_median_features.extend(result_list[i][5])
                Train_quant_25_features.extend(result_list[i][6])
                Train_quant_75_features.extend(result_list[i][7])
            X_train.extend(x_train)
            Y_train.extend(y_train)

        if randomize:
            if feat:
                X_train, Y_train, Train_cs_features, Train_spatial_features, Train_pattern_features, \
                Train_median_features, Train_quant_25_features, Train_quant_75_features = \
                    shuffle(X_train, Y_train, Train_cs_features, Train_spatial_features,Train_pattern_features,
                            Train_median_features, Train_quant_25_features, Train_quant_75_features, random_state=0)
            else:
                X_train, Y_train = shuffle(X_train, Y_train, random_state=0)

        print('Finished Generating : ', str(n_lag_days))
        sys.stdout.flush()

        if feat:
            return X_train, Y_train, Train_cs_features, Train_spatial_features, Train_pattern_features, \
                   Train_median_features, Train_quant_25_features, Train_quant_75_features
        else:
            return X_train, Y_train

    def generate_new_data(self, df, start, mid, stop):

        x = df.iloc[:, :-35][start:mid].to_numpy()
        y = df.iloc[:, :-35][mid:stop + 1].to_numpy()
        return x, y

    def split_df_train_test(self, df):

        X_train = list()
        X_test = list()
        Y_train = list()
        Y_test = list()

        start_time = self._config['data']['train_start']
        split_time = self._config['data']['train_stop']
        train_step = int(self._config['data']['train_window_size'])
        test_step = int(self._config['data']['test_window_size'])
        n_lag = int(self._config['data']['input_horizon'])
        n_lead = int(self._config['data']['output_horizon'])

        for i in range(n_lag, len(df)):
            end_ix = i + (n_lead - 1)
            # check if can create a pattern
            if end_ix >= len(df):
                break
            # retrieve input and output
            start_ix = i - n_lag

            if (i % train_step == 0) and (df.index[end_ix] <= datetime.datetime.strptime(split_time, '%Y-%m-%d %H:%M:%S')):
                if df.index[end_ix] > \
                        (datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(days=8)):
                    X, Y = self.generate_new_data(df, start_ix, i, end_ix)
                    X_train.append(X)
                    Y_train.append(Y)
            elif i % test_step == 0:
                X, Y = self.generate_new_data(df, start_ix, i, end_ix)
                X_test.append(X)
                Y_test.append(Y)

        return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

    def gen_and_save_new_data(self, df):

        train_dir = self._config['data']['train_path']
        val_dir = self._config['data']['val_path']
        n_lag_days = int(self._config['data']['input_horizon'])
        n_lead_days = int(self._config['data']['output_horizon'])

        x_train, y_train, x_test, y_test = self.split_df_train_test(df=df)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        train_step = self._config['data']['train_window_size']
        test_step = self._config['data']['test_window_size']

        np.save(os.path.join(train_dir, 'X_train_lag_' + str(n_lag_days) +
                             '_day_lead_' + str(n_lead_days) +
                             '_day_train_step_' + str(train_step) +
                             '_day_test_step_' + str(test_step) + '.npy'), x_train)

        np.save(os.path.join(train_dir, 'Y_train_lag_' + str(n_lag_days) +
                             '_day_lead_' + str(n_lead_days) +
                             '_day_train_step_' + str(train_step) +
                             '_day_test_step_' + str(test_step) + '.npy'), y_train)

        np.save(os.path.join(val_dir, 'X_test_lag_' + str(n_lag_days) +
                             '_day_lead_' + str(n_lead_days) +
                             '_day_train_step_' + str(train_step) +
                             '_day_test_step_' + str(test_step) + '.npy'), x_test)

        np.save(os.path.join(val_dir, 'Y_test_lag_' + str(n_lag_days) +
                             '_day_lead_' + str(n_lead_days) +
                             '_day_train_step_' + str(train_step) +
                             '_day_test_step_' + str(test_step) + '.npy'), y_test)


    def split_train_test(self, df, input_horizon, randomize=True):

        '''
        :param df: Dataframe containing all time series
        :param series_idx: index of series to split
        :param aggregate: if true function will aggregate the series and then return the train/test split
        :param randomize: if True shuffles the data
        :return: Train/Test split of data as numpy array
        '''

        feat = self._config.getboolean('data', 'features')

        if feat:
            X_train, Y_train, Train_cs_features, Train_spatial_features, Train_pattern_features, Train_median_features,\
            Train_quant_25_features, Train_quant_75_features = \
                self.generate_and_save_aggregated_train_test(df=df, input_horizon=input_horizon, randomize=randomize)
            return X_train, Y_train, Train_cs_features, Train_spatial_features, Train_pattern_features, \
                   Train_median_features, Train_quant_25_features, Train_quant_75_features
        else:
            X_train, Y_train = self.generate_and_save_aggregated_train_test(df=df, input_horizon=input_horizon,
                                                                            randomize=randomize)
            return X_train, Y_train




