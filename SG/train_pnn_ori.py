# -*- coding: utf-8 -*-
"""
Created on Mon May 20 2019
先识别一些已知的用户数据，然后再训练PNN得到网络后做预测
数据源为SG
@author: Zejian Zhou
"""

from typing import Dict, Any
from nilmtk import DataSet
import pandas as pd
import numpy as np
from pandas import DataFrame
import zejian_nilm3 as zz
import matplotlib.pyplot as plt
import pickle
from neupy import algorithms
from sklearn import metrics

#读入数据
water_heater = pd.read_excel ('../data/WaterHeater/WaterHeaterData.xlsx')
TV = pd.read_excel ('../data/TV/TVData.xlsx')
data_dic={'water heater': water_heater,
          'TV': TV}

# event detection
APPLIANCES = ['water heater', 'TV']
events: Dict[Any, Any] = {}
mains: Dict[Any, Any] = {}
for i in APPLIANCES:
    #分解数据
    mains_P = data_dic[i]['Active Power (W)']
    mains_I = data_dic[i]['Current (A)']
    mains_PF = data_dic[i]['Power factor']

    on_event_P, off_event_P = zz.get_activation(mains_P)
    on_event_I, off_event_I = zz.get_others(mains_I, on_event_P.index, off_event_P.index)
    on_event_PF, off_event_PF = zz.get_others(mains_I, on_event_P.index, off_event_P.index)

    df = {i + ' P on': on_event_P, i + ' P off': off_event_P,
        i + ' I on': on_event_I, i + ' I off': off_event_I,
        i + ' PF on': on_event_PF, i + ' PF off': off_event_PF}
    df2 = {i + ' P': mains_P, i + ' I': mains_I, i + ' PF': mains_PF}
    pass
    events.update(df)
    mains.update(df2)

# zz.plot_event(mains['water heater P'], events['water heater P on'], events['water heater P off'])


# 训练神经网络 PNN
pnn = algorithms.PNN(std=150, verbose=False)
# 准备training set 和 target set
count = 0
training_set = np.empty([1, 3])
target_set = np.empty([1, 1])
for i in APPLIANCES:
    # get active power data and reactive power data as a numpy
    P = np.array([events[i+' P on'].values]).T
    I = np.array([events[i+' I on'].values]).T
    PF = np.array([events[i+' PF on'].values]).T
    tr = np.concatenate((P, I, PF), axis=1)
    training_set = np.concatenate((training_set, tr), axis=0)
    targ = np.empty(tr.shape[0])
    targ[:] = count
    target_set = np.append(target_set, np.array([targ]).T, axis=0)
    count += 1
# 删掉第一个空元素
training_set = np.delete(training_set, 0, 0)
target_set = np.delete(target_set, 0, 0)

pnn.train(training_set, target_set)
pickle.dump(pnn, open('params/pnn.txt', 'wb'))
print(target_set.shape[0])