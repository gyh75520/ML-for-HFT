#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:18:07 2018

@author: howard
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, mean_squared_error
import os
import matplotlib.pyplot as plt


def file_name(dir):
    for root, dirs, files in os.walk(dir):
        return files


def get_files(TICK_FOLDER):
    dir = os.path.dirname(os.path.realpath('__file__'))
    dir += '/' + TICK_FOLDER[:-1]
    files = file_name(dir)
    return files


def read_csv_to_data(TICK_FOLDER):
    # 读取csv，分成 训练集 ，测试集
    data_trains = []
    data_tests = []
    files = get_files(TICK_FOLDER)
    print('len(files)', len(files) * 3 / 4)
    # 3/4 为训练集，其余为测试集
    for i in range(len(files)):
        if i < len(files) * 3 / 4:
            data_train_temp = pd.read_csv(TICK_FOLDER + files[i])
            data_trains.append(data_train_temp)
        else:
            data_test_temp = pd.read_csv(TICK_FOLDER + files[i])
            data_tests.append(data_test_temp)
    # data = pd.concat(datas)
    data_train = pd.concat(data_trains)
    data_test = pd.concat(data_tests)
    return data_train, data_test, data_tests


def write_ypred_to_csv(data_tests_arr, y_pred):
    # 测试集 + ypred to csv
    index = 0
    for i in range(len(data_tests_arr)):
        s1 = pd.DataFrame({'y_pred': y_pred[index:index + len(data_tests_arr[i])]})
        index += len(data_tests_arr[i])
        df = pd.merge(data_tests_arr[i], s1, left_index=True, right_index=True)
        df.to_csv('test_policy_accuracy' + str(i) + '.csv', index=False)


def make_table(data_test, y_pred):
    # 构造 real 和 prediction 表格
    values = [0, 1, -1]
    reals = []
    index_name = []
    pred_name = []
    for real_value in values:
        data_with_real_value = data_test[data_test['buy_hold_sell'] == real_value]
        real_value_pred = pd.DataFrame({'y_pred': y_pred[data_with_real_value.index]})
        preds = []
        for pred_value in values:
            pred = real_value_pred[real_value_pred['y_pred'] == pred_value].count()[0]
            # print (pred)
            preds.append(pred)
        reals.append(preds)
        index_name.append('real is ' + str(real_value))
        pred_name.append('predction is ' + str(real_value))

    # df = pd.DataFrame({'real is '+str(values[0]):reals[0],
    #                   'real is '+str(values[1]):reals[1],
    #                   'real is '+str(values[2]):reals[2],
    #                   })
    table = pd.DataFrame(reals)
    table.index = index_name
    table.columns = pred_name
    print(table)


def plot_compare(y_pred, y_test):
    plt.plot(np.array(y_pred[0:30]), 'o', color='red', label='prediction')
    plt.plot(np.array(y_test[0:30]), 'o', color='green', label='real')
    plt.xlabel('example')
    plt.ylabel('price')
    plt.legend(loc='best')
    plt.show()


def RandomForest_classifier_exp(data_train, data_test):
    print('---------------RandomForest_classifier_exp---------------')
    X_train = data_train.iloc[0:, 7:16]
    y_train = data_train['buy_hold_sell']
    X_test = data_test.iloc[0:, 7:16]
    y_test = data_test['buy_hold_sell']
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('accuracy_score:', accuracy_score(y_pred, y_test), '\n')
    make_table(data_test, y_pred)
    return y_pred


def RandomForest_regressor_exp(data_train, data_test):
    print('---------------RandomForest_regression_exp---------------')
    X_train = data_train.iloc[0:, 7:16]
    y_train = data_train['next_price']
    X_test = data_test.iloc[0:, 7:16]
    y_test = data_test['next_price']
    clf = RandomForestRegressor()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    plot_compare(y_pred, y_test)
    print('mean_squared_error:', mean_squared_error(y_pred, y_test), '\n')


def MlPClassifier_classifier_exp(data_train, data_test):
    print('---------------MlPClassifier_classifier_exp---------------')
    X_train = data_train.iloc[0:, 7:16]
    y_train = data_train['buy_hold_sell']
    X_test = data_test.iloc[0:, 7:16]
    y_test = data_test['buy_hold_sell']
    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('accuracy_score:', accuracy_score(y_pred, y_test), '\n')
    make_table(data_test, y_pred)
    return y_pred


def MLPRegressor_regressor_exp(data_train, data_test):
    print('---------------MLPRegressor_regressor_exp---------------')
    X_train = data_train.iloc[0:, 7:16]
    y_train = data_train['next_price']
    X_test = data_test.iloc[0:, 7:16]
    y_test = data_test['next_price']
    clf = MLPRegressor(hidden_layer_sizes=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    plot_compare(y_pred, y_test)
    print('mean_squared_error:', mean_squared_error(y_pred, y_test), '\n')


# TICK_FOLDER = 'single_tick//'
TICK_FOLDER = 'tick60036-1//'
data_train, data_test, data_tests_arr = read_csv_to_data(TICK_FOLDER)

RandomForest_classifier_exp(data_train, data_test)
# write_ypred_to_csv(data_tests_arr,y_pred)
# RandomForest_regressor_exp(data_train,data_test)
# MlPClassifier_classifier_exp(data_train,data_test)
# MLPRegressor_regressor_exp(data_train,data_test)
