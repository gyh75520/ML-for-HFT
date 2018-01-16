#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:22:49 2018

@author: howard
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import heapq
import os
np.set_printoptions(threshold=np.nan)
#TICK_FOLDER = 'single_tick//'
TICK_FOLDER = 'tick60036-1//'
POLICY_FOLDER = 'POLICY//'

TC = 0.001


def file_name(dir):
    for root, dirs, files in os.walk(dir):
        return files


hp = pd.DataFrame()


class Stock():
    def __init__(self):
        pass

    def init_stock(self, file):
        self.file = TICK_FOLDER + file
        self.peak, self.valley = [], []
        self.v, self.tv = [], []
        self.p, self.tp = [], []
        self.df = pd.read_csv(self.file, header=None, sep=',')
        self.price = np.array(self.df[9])
        self.get_extreme()

    def plot_price(self):
        plt.figure()
        plt.plot(self.price)
        plt.show()

    def get_extreme(self):
        tmp = self.price[0]
        for i in range(1, self.price.size - 1):
            if tmp > self.price[i] < self.price[i + 1]:
                self.valley.append((i, self.price[i]))
                tmp = self.price[i]
            elif tmp > self.price[i] == self.price[i + 1]:
                i += 1
            elif tmp < self.price[i] > self.price[i + 1]:
                self.peak.append((i, self.price[i]))
                tmp = self.price[i]
            elif tmp < self.price[i] == self.price[i + 1]:
                i += 1
            else:
                tmp = self.price[i]

        self.v = []
        self.tv = []
        for c in self.valley:
            self.v.append(c[1])
            self.tv.append(c[0])

        self.p, self.tp = [], []
        for c in self.peak:
            self.p.append(c[1])
            self.tp.append(c[0])

    def plot_extreme(self):
        plt.scatter(self.tv, self.v, alpha=0.5, color='r', marker='x')
        plt.scatter(self.tp, self.p, alpha=0.5, color='g', marker='.')
        plt.plot(self.price, alpha=0.3)
        plt.show()

    def policy(self):
        peak, valley = self.peak, self.valley
        if peak[0][0] < valley[0][0]:
            peak.pop(0)

        chance = []
        for i in range(min(len(valley), len(peak))):
            if (peak[i][1] - valley[i][1]) * 100 > TC * (peak[i][1] + valley[i][1]) * 100:
                chance.append((valley[i][0], peak[i][0]))

        policy = np.zeros(self.price.size, dtype=int)

        for c in chance:
            policy[c[0]], policy[c[1]] = +1, -1

        self.write_policy_to_csv(policy)
        self.write_next_price_to_csv()
#         print (policy)

        bp, bt = [], []
        sp, st = [], []
        for c in chance:
            bt.append(c[0])
            bp.append(self.price[c[0]])
            st.append(c[1])
            sp.append(self.price[c[1]])

        # plt.figure(figsize=(15, 5))
        #
        # plt.subplot(121)
        # plt.scatter(self.tv, self.v, alpha=0.5, color='r', marker='x')
        # plt.scatter(self.tp, self.p, alpha=0.5, color='g', marker='.')
        # plt.plot(self.price, alpha=0.3)
        #
        # plt.subplot(122)
        # plt.scatter(bt, bp, alpha=0.5, color='k', marker='o', edgecolor='white')
        # plt.scatter(st, sp, alpha=0.5, color='m', marker='x')
        # plt.plot(self.price, alpha=0.3)
        # plt.show()

#         plt.scatter(np.arange(self.price.size),np.array(policy), marker='x', color='b', alpha='0.5')
#         plt.show()
        print('transcation cnt:', len(chance))
        return policy

    def write_policy_to_csv(self, policy):
        s1 = pd.DataFrame({'buy_hold_sell': policy})
        #self.df = pd.concat([self.df, s1], axis=1, ignore_index=True)
        self.df = pd.merge(self.df, s1, left_index=True, right_index=True)
        self.df.to_csv(self.file, index=False)

    def write_next_price_to_csv(self):
        price = self.df.iloc[1:, 9:10]
        price.columns = ['next_price']
        price.index = price.index - 1
        self.df = pd.merge(self.df.iloc[:-1, :], price, left_index=True, right_index=True)
        self.df.to_csv(self.file, index=False)


dir = os.path.dirname(os.path.realpath(__file__))
dir += '/' + TICK_FOLDER[:-1]
print(dir)
files = file_name(dir)
ss = Stock()

for file in files:
    print(file)
    ss.init_stock(file)
    ss.policy()


# sorted(peak, key = lambda valley: valley[1], reverse = True)
