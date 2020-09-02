from data import Data
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from configparser import ConfigParser
import pickle
from baseline import Baseline

if __name__ == '__main__':

    '''data_obj = Data()
    df = data_obj.load_data()
    
    n_horizon = [1, 4,7]
    for n_days in n_horizon:
        data_obj.generate_and_save_aggregated_train_test(df, n_lag_days=n_days, n_lead_days=1)'''

    n_horizon = [1, 4, 7]

    baseline_approach = Baseline()

    for lag in n_horizon:
        baseline_approach.nearest_neighbour(n_lag_days=lag)






