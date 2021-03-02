import numpy as np
import pandas as pd
import os
import datetime
import sys
from configparser import ConfigParser
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler


def read_and_convert_dd(config, f_name, start, stop):
    data_dir = config['data']['path']

    # process and save as binary time series
    if not os.path.isfile(os.path.join(data_dir, 'dd.csv')):
        # preprocess
        df = pd.read_excel(os.path.join(data_dir, f_name), engine='openpyxl')
        df = df.drop(df[np.where(df["MeterTotal"] == 0, True, False)].index).reset_index(drop=True)
        df = df[(df.StartLocalTime >= start) & (df.EndLocalTime <= stop)].reset_index(drop=True)

        # create binary dataset df
        column_names = df.groupby(['ChargePointLabel', 'ConnectorType']).count().index.tolist()
        new_df = pd.DataFrame(columns=column_names,
                              index=pd.date_range(start=start, end=stop, freq='15min')).fillna(0)
        new_df.columns = pd.MultiIndex.from_tuples(new_df.columns, names=['ChargePointLabel', 'ConnectorType'])

        # fill binary values
        index = new_df.index
        columns = new_df.columns
        last_idx = 0
        for i in range(len(df)):
            start_time = df.iloc[i, 0]
            stop_time = df.iloc[i, 1]
            c_label = df.iloc[i, 4]
            c_type = df.iloc[i, 8]

            if i % 10000 == 0:
                print(i, sep=' ', end=' ')

            c_in = 0
            for c in range(len(columns)):
                if columns[c][0] == c_label and columns[c][1] == c_type:
                    c_in = c

            for idx in range(last_idx, len(index)):
                if start_time > index[idx]:
                    last_idx = idx
                    continue
                if index[idx] < stop_time:
                    new_df.iloc[idx, c_in] = 1
                    continue
                break

        new_df.to_csv(os.path.join(data_dir, 'dd.csv'))

    # read, add time features and return df
    dd_df = pd.read_csv(os.path.join(data_dir, 'dd.csv'), index_col=[0], header=[0, 1])
    dd_df.index = pd.to_datetime(dd_df.index)

    dd_df['month'] = dd_df.index.month
    dd_df['weekday'] = dd_df.index.dayofweek
    dd_df['hour'] = dd_df.index.hour
    dd_df['minute'] = dd_df.index.minute
    dd_df = pd.get_dummies(dd_df,
                            prefix=['month', 'weekday', 'hour', 'minute'],
                            columns=['month', 'weekday', 'hour', 'minute'])
    return dd_df

def gen_pattern_features_(config, df):
    train_df = df.loc[config['data']['train_start'] : config['data']['train_stop']].iloc[:, :-47]
    daily_usage = train_df.groupby([train_df.index.hour, train_df.index.minute]).sum()
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(daily_usage)
    scaled_df = pd.DataFrame(scaled_features, index=daily_usage.index, columns=daily_usage.columns)
    return scaled_df

def generate_data_(series, df, pattern_feature, start, stop):

    d = np.expand_dims(series.to_numpy()[start:stop], axis=1)
    idx = series.name
    pattern_feat = pattern_feature[idx].to_numpy()
    time_feat = df.iloc[:, -47:].to_numpy()[start:stop]
    data = np.concatenate([d, time_feat], axis=1)
    return data, pattern_feat

def split_series_train_test_(config, series, df, pattern_feature):
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
    train_pattern_features = list()
    test_pattern_features = list()

    start_time = config['data']['train_start']
    split_time = config['data']['train_stop']
    train_step = int(config['data']['train_window_size'])
    test_step = int(config['data']['test_window_size'])
    n_lag = int(config['data']['input_horizon'])  # *96 when lag value is in days
    n_lead = int(config['data']['output_horizon']) # *96 when lead is in days
    feat = config.getboolean('data', 'features')

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
                    data, pattern_feat = generate_data_(series, df, pattern_feature, start_ix, i)
                    X_train.append(data.tolist())
                    train_pattern_features.append(pattern_feat.tolist())
                else:
                    X_train.append(series.tolist()[start_ix:i])
                Y_train.append(series.tolist()[i:(end_ix + 1)])

        elif i%test_step == 0:

            if feat:
                data, pattern_feat = generate_data_(series, df, pattern_feature, start_ix, i)
                X_test.append(data.tolist())
                test_pattern_features.append(pattern_feat.tolist())
            else:
                X_test.append(series.tolist()[start_ix:i])
            Y_test.append(series.tolist()[i:(end_ix + 1)])

    if feat:
        X_train, Y_train, train_pattern_features = \
            shuffle(X_train, Y_train, train_pattern_features, random_state=0)
        X_test, Y_test, test_pattern_features = \
            shuffle(X_test, Y_test, test_pattern_features, random_state=0)
    else:
        X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
        X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

    if feat:
        return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test), \
               np.array(train_pattern_features), np.array(test_pattern_features)
    else:
        return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


def gen_and_save_(config, df):
    train_dir = config['data']['train_path']
    val_dir = config['data']['val_path']
    n_lag_days = int(config['data']['input_horizon'])
    n_lead_days = int(config['data']['output_horizon'])
    feat = config.getboolean('data', 'features')

    X_train = list()
    Y_train = list()
    X_test = list()
    Y_test = list()
    Train_pattern_features = list()
    Test_pattern_features = list()

    pattern_feature = gen_pattern_features_(config, df)
    print(pattern_feature.shape)

    for series_idx in range(df.shape[1] - 47):  # subtract last 9 time feature columns
        if feat:
            x_train, y_train, x_test, y_test, \
            train_pattern_features, test_pattern_features = split_series_train_test_(config, df.iloc[:, series_idx],
                                                                                         df, pattern_feature)
            Train_pattern_features.extend(train_pattern_features.tolist())
            Test_pattern_features.extend(test_pattern_features.tolist())
        else:
            x_train, y_train, x_test, y_test = split_series_train_test_(config, df.iloc[:, series_idx],
                                                                            df, pattern_feature)
        X_train.extend(x_train.tolist())
        Y_train.extend(y_train.tolist())
        X_test.extend(x_test.tolist())
        Y_test.extend(y_test.tolist())

    # shuffle and save the data
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    if feat:
        Train_pattern_features = np.array(Train_pattern_features)
        Test_pattern_features = np.array(Test_pattern_features)

        X_train, Y_train, Train_pattern_features = shuffle(X_train, Y_train, Train_pattern_features, random_state=0)
        X_test, Y_test, Test_pattern_features = shuffle(X_test, Y_test, Test_pattern_features, random_state=0)
    else:
        X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
        X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

    train_step = config['data']['train_window_size']
    test_step = config['data']['test_window_size']

    np.save(os.path.join(train_dir, 'X_train_lag_' + str(n_lag_days) +
                         '_day_lead_' + str(n_lead_days) +
                         '_day_train_step_' + str(train_step) +
                         '_day_test_step_' + str(test_step) + '.npy'), X_train)

    np.save(os.path.join(train_dir, 'Y_train_lag_' + str(n_lag_days) +
                         '_day_lead_' + str(n_lead_days) +
                         '_day_train_step_' + str(train_step) +
                         '_day_test_step_' + str(test_step) + '.npy'), Y_train)

    np.save(os.path.join(val_dir, 'X_test_lag_' + str(n_lag_days) +
                         '_day_lead_' + str(n_lead_days) +
                         '_day_train_step_' + str(train_step) +
                         '_day_test_step_' + str(test_step) + '.npy'), X_test)

    np.save(os.path.join(val_dir, 'Y_test_lag_' + str(n_lag_days) +
                         '_day_lead_' + str(n_lead_days) +
                         '_day_train_step_' + str(train_step) +
                         '_day_test_step_' + str(test_step) + '.npy'), Y_test)

    if feat:

        np.save(os.path.join(train_dir, 'Train_pattern_features_lag_' + str(n_lag_days) +
                             '_day_lead_' + str(n_lead_days) +
                             '_day_train_step_' + str(train_step) +
                             '_day_test_step_' + str(test_step) + '.npy'), Train_pattern_features)

        np.save(os.path.join(val_dir, 'Test_pattern_features_lag_' + str(n_lag_days) +
                             '_day_lead_' + str(n_lead_days) +
                             '_day_train_step_' + str(train_step) +
                             '_day_test_step_' + str(test_step) + '.npy'), Test_pattern_features)

    print('Finished Generating : ', str(n_lag_days))
    sys.stdout.flush()


def split_train_test_(config, df):
    '''
    :param df: Dataframe containing all time series
    :param series_idx: index of series to split
    :param aggregate: if true function will aggregate the series and then return the train/test split
    :param randomize: if True shuffles the data
    :return: Train/Test split of data as numpy array
    '''

    input_horizon = int(config['data']['input_horizon'])
    output_horizon = int(config['data']['output_horizon'])

    train_step = config['data']['train_window_size']
    test_step = config['data']['test_window_size']
    train_path = config['data']['train_path']
    val_path = config['data']['val_path']
    feat = config.getboolean('data', 'features')

    if not os.path.isfile(os.path.join(train_path, 'X_train_lag_' + str(input_horizon) +
                                                   '_day_lead_' + str(output_horizon) +
                                                   '_day_train_step_' + str(train_step) +
                                                   '_day_test_step_' + str(test_step) + '.npy')):
        gen_and_save_(config, df=df)

    X_train = np.load(os.path.join(train_path, 'X_train_lag_' + str(input_horizon) +
                                   '_day_lead_' + str(output_horizon) +
                                   '_day_train_step_' + str(train_step) +
                                   '_day_test_step_' + str(test_step) + '.npy'))

    Y_train = np.load(os.path.join(train_path, 'Y_train_lag_' + str(input_horizon) +
                                   '_day_lead_' + str(output_horizon) +
                                   '_day_train_step_' + str(train_step) +
                                   '_day_test_step_' + str(test_step) + '.npy'))

    X_test = np.load(os.path.join(val_path, 'X_test_lag_' + str(input_horizon) +
                                  '_day_lead_' + str(output_horizon) +
                                  '_day_train_step_' + str(train_step) +
                                  '_day_test_step_' + str(test_step) + '.npy'))

    Y_test = np.load(os.path.join(val_path, 'Y_test_lag_' + str(input_horizon) +
                                  '_day_lead_' + str(output_horizon) +
                                  '_day_train_step_' + str(train_step) +
                                  '_day_test_step_' + str(test_step) + '.npy'))

    if feat:

        Train_pattern_features = np.load(os.path.join(train_path, 'Train_pattern_features_lag_' + str(input_horizon) +
                                                      '_day_lead_' + str(output_horizon) +
                                                      '_day_train_step_' + str(train_step) +
                                                      '_day_test_step_' + str(test_step) + '.npy'))

        Test_pattern_features = np.load(os.path.join(val_path, 'Test_pattern_features_lag_' + str(input_horizon) +
                                                     '_day_lead_' + str(output_horizon) +
                                                     '_day_train_step_' + str(train_step) +
                                                     '_day_test_step_' + str(test_step) + '.npy'))

        return X_train, Y_train, X_test, Y_test, \
               Train_pattern_features, Test_pattern_features
    else:
        return X_train, Y_train, X_test, Y_test