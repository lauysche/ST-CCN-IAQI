# coding:utf-8
import joblib
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    data = None
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        #data = joblib.load(f)
    return data


def get_ids_for_tvt(step):
    train_ids = []
    valid_ids = []
    test_ids = []
    #days_in_months = [31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30-1]  # May to April
    start_id = 0
    #for i in range(len(days_in_months)):
    #days = days_in_months[i]
    #days=51
    #days=370
    days=34*5
    split_id_0 = start_id
    #split_id_1 = start_id + int(days * 24 * 0.6)
    split_id_1 = start_id + int((days-1) * 24 * (5/7))
    split_id_2 = start_id + int((days-1) * 24 * (6/7))
    split_id_3 = (days-1)*24
    split_id_3 = (days-1)*24-(step-1)
    #split_id_3 = (days - 2) * 24 - (step - 1)
    #split_id_3 = 17736
    #split_id_3 = 8856
    #split_id_3 = 1200
    #split_id_3 = start_id + int(days * 24)
    train_ids.extend(np.arange(split_id_0, split_id_1, 1))
    valid_ids.extend(np.arange(split_id_1, split_id_2, 1))
    test_ids.extend(np.arange(split_id_2, split_id_3, 1))
    #start_id += int(days * 24)
    return train_ids, valid_ids, test_ids


def load_data(f_x, f_y, step):
    x = load_pickle(f_x)
    y = load_pickle(f_y)
    y = np.array(y[:, np.newaxis])
    if len(x.shape) == 3:
        ss = preprocessing.StandardScaler()
        for i in range(x.shape[-1]):
            ss.fit(x[:, :, i])
            x[:, :, i] = ss.transform(x[:, :, i])
    train_ids, valid_ids, test_ids = get_ids_for_tvt(step)
    x_train = x[train_ids]
    y_train = y[train_ids]
    x_valid = x[valid_ids]
    y_valid = y[valid_ids]
    x_test = x[test_ids]
    y_test = y[test_ids]
    '''
    print(x_train.shape)
    for i in range(len(x_train)):
        print(x_train[i:i+1][i])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train = pd.Series(x_train[:,0]).values.reshape(-1,1)
    #x_train = scaler.fit_transform(x_train.reshape(-1, 1))
    y_train = scaler.transform(pd.Series(y_train).values.reshape(-1, 1))
    x_valid = scaler.fit_transform(pd.Series(x_valid).values.reshape(-1, 1))
    y_valid = scaler.transform(pd.Series(y_valid).values.reshape(-1, 1))
    x_test = scaler.fit_transform(pd.Series(x_test).values.reshape(-1, 1))
    y_test = scaler.transform(pd.Series(y_test).values.reshape(-1, 1))
    '''
    print('x_shape: {}  y_shape: {}\nx_train_shape: {}  y_train_shape: {}  x_valid_shape: {}  y_valid_shape: {}  x_test_shape: {}  y_test_shape: {}\n'
          .format(x.shape, y.shape, x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape, y_test.shape))
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_param_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num
