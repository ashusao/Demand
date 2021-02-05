import numpy as np
from configparser import ConfigParser
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support

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

    def nearest_neighbour(self, X_train, Y_train, X_test, Y_test):

        model_path = self._config['model']['knn_path']
        input_horizon = self._input_horizon
        output_horizon = self._output_horizon

        loaded = False

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
        precision, recall, f1, _ = precision_recall_fscore_support(Y_test.ravel(), pred.ravel(), average=None)

        if not loaded:
            pickle.dump(neigh, open(model_file, 'wb'))

        return precision, recall, f1

    def support_vector_classifier(self, X_train, Y_train, X_test, Y_test):

        model_path = self._config['model']['svm_path']
        input_horizon = self._input_horizon
        output_horizon = self._output_horizon

        loaded = False
        n_estimators = 10

        model_file = os.path.join(model_path, 'svm_complte_lag_' + str(input_horizon) +
                                             '_lead_' + str(output_horizon) +
                                             '_train_step_' + str(self._config['data']['train_window_size']) +
                                             '_est_step_' + str(self._config['data']['test_window_size']) +
                                             '.pkl')
        if os.path.isfile(model_file):
            clf = pickle.load(open(model_file, 'rb'))
            loaded = True

        if not loaded:
            clf = OneVsRestClassifier(SVC(kernel='linear', gamma='scale'), n_jobs=5)
            clf.fit(X_train, Y_train)

        pred = clf.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(Y_test.ravel(), pred.ravel(), average=None)

        if not loaded:
            pickle.dump(clf, open(model_file, 'wb'))

        return precision, recall, f1

    def random_forest(self, X_train, Y_train, X_test, Y_test):

        model_path = self._config['model']['rf_path']
        input_horizon = self._input_horizon
        output_horizon = self._output_horizon

        loaded = False

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
        precision, recall, f1, _ = precision_recall_fscore_support(Y_test.ravel(), pred.ravel(), average=None)

        if not loaded:
            pickle.dump(clf, open(model_file, 'wb'))

        return precision, recall, f1

    def historical_average(self, X_train, Y_train, X_test, Y_test):
        pass

    def load_model(self, algo):

        clf = None
        if algo == 'knn':
            model_path = self._config['model']['knn_path']
            model_file = os.path.join(model_path, 'knn_complte_lag_' + str(self._input_horizon) +
                                      '_lead_' + str(self._output_horizon) +
                                      '_train_step_' + str(self._config['data']['train_window_size']) +
                                      '_est_step_' + str(self._config['data']['test_window_size']) +
                                      '.pkl')
            clf = pickle.load(open(model_file, 'rb'))
        elif algo == 'rf':
            model_path = self._config['model']['rf_path']
            model_file = os.path.join(model_path, 'rf_complte_lag_' + str(self._input_horizon) +
                                      '_lead_' + str(self._output_horizon) +
                                      '_train_step_' + str(self._config['data']['train_window_size']) +
                                      '_est_step_' + str(self._config['data']['test_window_size']) +
                                      '.pkl')
            clf = pickle.load(open(model_file, 'rb'))
        elif algo == 'svm':
            model_path = self._config['model']['svm_path']
            model_file = os.path.join(model_path, 'svm_complte_lag_' + str(self._input_horizon) +
                                      '_lead_' + str(self._output_horizon) +
                                      '_train_step_' + str(self._config['data']['train_window_size']) +
                                      '_est_step_' + str(self._config['data']['test_window_size']) +
                                      '.pkl')
            clf = pickle.load(open(model_file, 'rb'))

        return clf

    def write_to_csv(self, result_file, row):

        if not os.path.isfile(result_file):
            header = ['n_train', 'n_test', 'input_horizon', 'output_horizon', 'threshold',
                      'prec_0', 'prec_1', 'rec_0', 'rec_1', 'F1_0', 'F1_1', 'comment']

            with open(result_file, "a+", newline='') as f:
                csv_writer = csv.writer(f, delimiter=',')
                csv_writer.writerow(header)
                csv_writer.writerow(row)
        else:
            with open(result_file, "a+", newline='') as f:
                csv_writer = csv.writer(f, delimiter=',')
                csv_writer.writerow(row)

    def eval_test_set(self, n_train, X, Y):

        prec_0 = list()
        prec_1 = list()
        rec_0 = list()
        rec_1 = list()
        f1_0 = list()
        f1_1 = list()

        result_path = self._config['result']['path']
        comment = self._config['result']['comment']

        algo = self._config['train']['algo']
        clf = self.load_model(algo)

        for i in range(len(X)):
            pred = clf.predict(X[i])
            precision, recall, f1, _ = precision_recall_fscore_support(Y[i].ravel(), pred.ravel(), average=None)

            prec_0.append(precision[0])
            prec_1.append(precision[1])
            rec_0.append(recall[0])
            rec_1.append(recall[1])
            f1_0.append(f1[0])
            f1_1.append(f1[1])

            result_row = [n_train, X[i].shape[0], self._input_horizon, self._output_horizon, 0,
                          precision[0], precision[1], recall[0], recall[1], f1[0], f1[1], comment]

            result_file = os.path.join(result_path, 'test_set_' + str(i + 1) + '.csv')
            self.write_to_csv(result_file, result_row)

        result_file = os.path.join(result_path, 'avg_test_set.csv')

        prec_0 = np.array(prec_0)
        prec_1 = np.array(prec_1)
        rec_0 = np.array(rec_0)
        rec_1 = np.array(rec_1)
        f1_0 = np.array(f1_0)
        f1_1 = np.array(f1_1)

        result_row = [n_train, X[i].shape[0], self._input_horizon, self._output_horizon, np.mean(prec_0), np.mean(prec_1),
                      np.mean(rec_0), np.mean(rec_1), np.mean(f1_0), np.mean(f1_1), comment]

        self.write_to_csv(result_file, result_row)

    def log_result(self, n_train, n_test, prec, rec, f1, comment):

        result_row = [self._input_horizon, self._output_horizon,
                      n_train, n_test, prec[0], prec[1], rec[0], rec[1], f1[0], f1[1], comment]

        # save result in csv
        result_path = self._config['result']['path']
        algo = self._config['train']['algo']

        if algo == 'knn':
            result_file = os.path.join(result_path, 'knn_new.csv')
        elif algo == 'rf':
            result_file = os.path.join(result_path, 'rf_new.csv')
        elif algo == 'svm':
            result_file = os.path.join(result_path, 'svm.csv')

        if not os.path.isfile(result_file):
            header = ['Input_Horizon', 'Output_Horizon', 'n_train', 'n_test',
                      'prec_0', 'prec_1', 'rec_0', 'rec_1', 'f1_0', 'f1_1', 'comment']
            with open(result_file, "a+", newline='') as f:
                csv_writer = csv.writer(f, delimiter=',')
                csv_writer.writerow(header)
                csv_writer.writerow(result_row)
        else:
            with open(result_file, "a+", newline='') as f:
                csv_writer = csv.writer(f, delimiter=',')
                csv_writer.writerow(result_row)





