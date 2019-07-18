# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 2019
先识别一些已知的用户数据，然后再训练PNN得到网络后做预测
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
iawe.set_window(start='06-07-2013', end='06-15-2013')
elec = iawe.buildings[1].elec

# event detection
APPLIANCES = ['fridge', 'clothes iron', 'computer']
events: Dict[Any, Any] = {}
mains: Dict[Any, Any] = {}
for i in APPLIANCES:
    print(elec[i].available_columns())
    mains_activ = next(elec[i].load(ac_type='active'))['power', 'active'].fillna(0).dropna()
    mains_reactiv = next(elec[i].load(ac_type='reactive'))['power', 'reactive'].fillna(0).dropna()

    on_event_active, off_event_active = zz.get_activation(mains_activ)
    on_event_reactive, off_event_reactive = zz.get_others(mains_reactiv, on_event_active.index, off_event_active.index)

    df = {i + ' active on': on_event_active, i + ' active off': off_event_active,
        i + ' reactive on': on_event_reactive, i + ' reactive off': off_event_reactive}
    df2 = {i + ' active power': mains_activ, i + ' reactive power': mains_reactiv}
    pass
    events.update(df)
    mains.update(df2)

#zz.plot_event(mains['computer active power'], events['computer active on'], events['computer active off'])


# 训练神经网络 PNN
pnn = algorithms.PNN(std=150, verbose=False)
# 准备training set 和 target set
count = 0
training_set = np.empty([1,2])
target_set = np.empty([1, 1])
for i in APPLIANCES:
    # get active power data and reactive power data as a numpy
    activ = np.array([events[i+' active on'].values]).T
    reactiv = np.array([events[i+' reactive on'].values]).T
    tr = np.concatenate((activ, reactiv), axis=1)
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