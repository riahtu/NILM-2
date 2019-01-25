# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 2019
先识别一些已知的用户数据，然后再训练PNN得到网络后做预测
online feed data
@author: Zejian Zhou
"""

from typing import Dict, Any
from nilmtk import DataSet
import pandas as pd
import numpy as np
from pandas import DataFrame
import zejian_nilm2 as zz
import matplotlib.pyplot as plt
import pickle
from neupy import algorithms
from sklearn import metrics

iawe = DataSet('data/iawe.h5')
# fridge和computer的数据从7-01开始不能用了， 7-20 开始好了，后面都不能用，21号能用一天
iawe.set_window(start='6-20-2013', end='6-21-2013')
elec = iawe.buildings[1].elec

# 做一个合成的数据
APPLIANCES = ['fridge', 'computer']
events: Dict[Any, Any] = {}
mains: Dict[Any, Any] = {}
mains_activ=0
mains_reactiv=0
start_time=next(elec['fridge'].load(ac_type='active'))['power', 'active'].index[0]
delay_time=500
tspan=[start_time, start_time + pd.Timedelta(seconds=delay_time)]
# 读入神经网络 PNN
pnn = pickle.load(open('params/pnn.txt', "rb"))

for i in APPLIANCES:
    #print(elec[i].available_columns())
    mains_activ += next(elec[i].load(ac_type='active'))['power', 'active'].fillna(0)
    mains_reactiv += next(elec[i].load(ac_type='reactive'))['power', 'reactive'].fillna(0)
# 如果遇到丢掉的数据就用前后来非零来代替 TODO: fix it with a better algorithm to handle package loss
mains_activ=mains_activ.fillna(method='ffill')
mains_reactiv=mains_reactiv.fillna(method='ffill')
# event detection 和 active/ reactive power 计算
on_event_active, off_event_active = zz.get_activation(mains_activ)
# 如果每有event就出去重新来
on_event_reactive, off_event_reactive = zz.get_others(mains_reactiv, on_event_active.index, off_event_active.index)
#zz.plot_event(mains['computer active power'], events['computer active on'], events['computer active off'])


# 准备training set 和 target set
training_set = np.empty([1, 2])
target_set = np.empty([1, 1])
# get active power data and reactive power data as a numpy
activ = np.array([on_event_active.values]).T
reactiv = np.array([on_event_reactive.values]).T
tr = np.concatenate((activ, reactiv), axis=1)
training_set = np.concatenate((training_set, tr), axis=0)
# 删掉第一个空元素
#TODO: 别忘了检测关
training_set = np.delete(training_set, 0, 0)
y_predicted = pnn.predict(training_set)
pass
# 带入测试
# print(y_predicted-target_set)
    # print(target_set.shape[0])