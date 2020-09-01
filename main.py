from data import Data
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from configparser import ConfigParser
import pickle

if __name__ == '__main__':

    '''data_obj = Data()
    df = data_obj.load_data()

    n_horizon = [1,4,7]
    for n_days in n_horizon:
        data_obj.generate_and_save_aggregated_train_test(df, n_lag_days=n_days, n_lead_days=1)'''

    config = ConfigParser()
    config.read('config.ini')

    npy_path = config['data']['npy_path']
    model_path = config['model']['path']

    X_train = np.load(os.path.join(npy_path, 'X_train_lag_1_day.npy'))
    Y_train = np.load(os.path.join(npy_path, 'Y_train_lag_1_day.npy'))
    X_test = np.load(os.path.join(npy_path, 'X_test_lag_1_day.npy'))
    Y_test = np.load(os.path.join(npy_path, 'Y_test_lag_1_day.npy'))

    neigh = KNeighborsClassifier(n_neighbors=1, metric='matching', algorithm='ball_tree', n_jobs=-1)
    neigh.fit(X_train, Y_train)
    predictions = neigh.predict(X_test)
    print(balanced_accuracy_score(Y_test.ravel(), predictions.ravel()))
    print(f1_score(Y_test.ravel(), predictions.ravel()))

    pickle.dump(neigh, open(os.path.join(model_path, 'knn_complte_lag_1.pkl'), 'wb'))




