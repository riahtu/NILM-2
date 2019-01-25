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
#zz.plot_event(mains_activ, on_event_active, off_event_active)


appliance_on = zz.predict_appliance(pnn, on_event_active, on_event_reactive)
appliance_off = zz.predict_appliance(pnn, off_event_active, off_event_reactive, flag_off=True)

zz.plot_appliance(mains_activ, appliance_on, appliance_off, on_event_active, off_event_active)

f_on = open('res/app_on.txt', 'w')
f_off = open('res/app_off.txt', 'w')
if len(on_event_active) != 0:
    for k in range(appliance_on.size):
        if appliance_on[k] == 0:
            print(on_event_active.index[k], 'fridge on')
            print(on_event_active.index[k], 'fridge on', file=f_on)
        elif appliance_on[k] == 2:
            print(on_event_active.index[k], 'computer on')
            print(on_event_active.index[k], 'computer on', file=f_on)

if len(off_event_active) != 0:
    for k in range(appliance_off.size):
        if appliance_off[k] == 0:
            print(off_event_active.index[k], 'fridge off')
            print(off_event_active.index[k], 'fridge off', file=f_off)
        elif appliance_off[k] == 2:
            print(off_event_active.index[k], 'computer off')
            print(off_event_active.index[k], 'computer off', file=f_off)
f_on.close()
f_off.close()
pass
