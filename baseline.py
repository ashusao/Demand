import numpy as np
from configparser import ConfigParser
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

import csv
import os
import sys

from data import Data

class Baseline:

    def __init__(self):
        self._config = ConfigParser()
        self._config.read('config.ini')

    def nearest_neighbour(self, n_lag_days=1, n_lead_days=1, k=1, metric='matching', algo='ball_tree'):

        model_path = self._config['model']['path']
        result_path = self._config['result']['path']

        # load data
        data_obj = Data()
        X_train, Y_train, X_test, Y_test = data_obj.load_npy(n_lag_days)

        neigh = KNeighborsClassifier(n_neighbors=k, metric=metric, algorithm=algo, n_jobs=-1)
        neigh.fit(X_train, Y_train)
        predictions = neigh.predict(X_test)
        acc = balanced_accuracy_score(Y_test.ravel(), predictions.ravel())
        f1 = f1_score(Y_test.ravel(), predictions.ravel())

        result_row = ['Complete', n_lag_days, 1, 1, 1, acc, f1]

        # save result in csv
        result_file = os.path.join(result_path, 'knn.csv')

        if not os.path.isfile(result_file):
            header = ['Data', 'Input_Horizon', 'Output_Horizon', 'train_step_size', 'test_step_size', 'balanced_acurracy', 'f1_score']
            with open(result_file , "a+", newline='') as f:
                csv_writer = csv.writer(f, delimiter=',')
                csv_writer.writerow(header)
                csv_writer.writerow(result_row)
        else:
            with open(result_file , "a+", newline='') as f:
                csv_writer = csv.writer(f, delimiter=',')
                csv_writer.writerow(result_row)


        pickle.dump(neigh, open(os.path.join(model_path, 'knn_complte_lag_' + str(n_lag_days) +'.pkl'), 'wb'))

        print('Input Horizon: ' + str(n_lag_days), ', Bal Accuracy: ' + str(acc) + ', F1: ' + str (f1))
        sys.stdout.flush()



