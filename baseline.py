import numpy as np
from configparser import ConfigParser
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import csv
import os
import sys
import pickle

from data import Data

class Baseline:

    def __init__(self):
        self._config = ConfigParser()
        self._config.read('config.ini')
        self._input_horizon = int(self._config['data']['input_horizon'])
        self._output_horizon = int(self._config['data']['output_horizon'])

    def nearest_neighbour(self, X_train, Y_train, X_test, Y_test, aggregate=True):

        model_path = self._config['model']['knn_path']
        input_horizon = self._input_horizon
        output_horizon = self._output_horizon

        loaded = False
        neigh = None
        model_file = None

        if aggregate: # load saved model
            model_file = os.path.join(model_path, 'knn_complte_lag_' + str(input_horizon) +
                                                 '_lead_' + str(output_horizon) +
                                                 '_train_step_' + str(self._config['data']['train_window_size']) +
                                                 '_est_step_' + str(self._config['data']['test_window_size']) +
                                                 '.pkl')
            if os.path.isfile(model_file):
                with open(model_file, "rb") as f:
                    neigh = pickle.load(f)
                loaded = True

        if not loaded:
            neigh = KNeighborsClassifier(n_neighbors=1, metric='matching', n_jobs=-1)
            neigh.fit(X_train, Y_train)

        pred = neigh.predict(X_test)
        acc = balanced_accuracy_score(Y_test.ravel(), pred.ravel())
        f1 = f1_score(Y_test.ravel(), pred.ravel(), average=None)
        cm = confusion_matrix(Y_test.ravel(), pred.ravel())

        if not loaded and aggregate:
            pickle.dump(clf, open(model_file, 'wb'))

        return acc, f1, cm

    def random_forest(self, X_train, Y_train, X_test, Y_test, aggregate=True):

        model_path = self._config['model']['rf_path']
        input_horizon = self._input_horizon
        output_horizon = self._output_horizon

        loaded = False
        clf = None
        model_file = None

        if aggregate:
            model_file = os.path.join(model_path, 'rf_complte_lag_' + str(input_horizon) +
                                                 '_lead_' + str(output_horizon) +
                                                 '_train_step_' + str(self._config['data']['train_window_size']) +
                                                 '_est_step_' + str(self._config['data']['test_window_size']) +
                                                 '.pkl')
            if os.path.isfile(model_file):
                clf = pickle.load(open(model_file, 'rb'))
                loaded = True

        if not loaded:
            clf = RandomForestClassifier(random_state=0, n_jobs=-1)
            clf.fit(X_train, Y_train)

        pred = clf.predict(X_test)
        acc = balanced_accuracy_score(Y_test.ravel(), pred.ravel())
        f1 = f1_score(Y_test.ravel(), pred.ravel(), average=None)
        cm = confusion_matrix(Y_test.ravel(), pred.ravel())

        if not loaded and aggregate:
            pickle.dump(clf, open(model_file, 'wb'))

        return acc, f1, cm

    def log_result(self, data_type, n_train, n_test, accuracy, f1):

        result_row = [data_type, self._input_horizon, self._output_horizon,
                      self._config['data']['train_window_size'],
                      self._config['data']['test_window_size'],
                      n_train, n_test, accuracy, f1[0], f1[1]]

        # save result in csv
        result_path = self._config['result']['path']
        algo = self._config['train']['algo']

        if algo == 'knn':
            result_file = os.path.join(result_path, 'knn_new.csv')
        elif algo == 'rf':
            result_file = os.path.join(result_path, 'rf_new.csv')

        if not os.path.isfile(result_file):
            header = ['Data', 'Input_Horizon', 'Output_Horizon', 'train_step_size', 'test_step_size',
                      'n_train', 'n_test', 'balanced_acurracy', 'f1_0', 'f1_1']
            with open(result_file, "a+", newline='') as f:
                csv_writer = csv.writer(f, delimiter=',')
                csv_writer.writerow(header)
                csv_writer.writerow(result_row)
        else:
            with open(result_file, "a+", newline='') as f:
                csv_writer = csv.writer(f, delimiter=',')
                csv_writer.writerow(result_row)





