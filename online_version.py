# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 2019
先识别一些已知的用户数据，然后再训练PNN得到网络后做预测
online feed data, 基础版本，问题是现在的算法浪费了比较多的数据
@author: Zejian Zhou
"""
# TODO: 别忘了调整参数
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

f_on = open('res/test_app_on.txt', 'w')
f_off = open('res/test_app_off.txt', 'w')
for j in range(2, 200):
    for i in APPLIANCES:
        #print(elec[i].available_columns())
        mains_activ += next(elec[i].load(ac_type='active'))['power', 'active'].fillna(0)[tspan[0]:tspan[1]]
        mains_reactiv += next(elec[i].load(ac_type='reactive'))['power', 'reactive'].fillna(0)[tspan[0]:tspan[1]]
    # 如果遇到丢掉的数据就用前后来非零来代替 TODO: fix it with a better algorithm to handle package loss
    mains_activ=mains_activ.fillna(method='ffill')
    mains_reactiv=mains_reactiv.fillna(method='ffill')
    # event detection 和 active/ reactive power 计算
    on_event_active, off_event_active = zz.get_activation(mains_activ)
    # 如果每有event就出去重新来
    if len(on_event_active) == 0 and len(off_event_active) == 0:
        mains_activ = 0
        mains_reactiv = 0
        #print(tspan[0], tspan[1], 'no event')
        # 增加时间
        tspan = [start_time + (j-1)*pd.Timedelta(seconds=delay_time), start_time + j*pd.Timedelta(seconds=delay_time)]
        continue
    if len(on_event_active) == 0:
        on_in=[]
    else:
        on_in=on_event_active.index
    if len(off_event_active) == 0:
        off_in=[]
    else:
        off_in=off_event_active.index
    on_event_reactive, off_event_reactive = zz.get_others(mains_reactiv, on_in, off_in)
    #zz.plot_event(mains['computer active power'], events['computer active on'], events['computer active off'])

    # do prediction
    if len(on_event_active) != 0:
        appliance_on = zz.predict_appliance(pnn, on_event_active, on_event_reactive)
    if len(off_event_active) != 0:
        appliance_off = zz.predict_appliance(pnn, off_event_active, off_event_reactive, flag_off=True)

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


    # 增加时间
    tspan = [start_time + (j - 1) * pd.Timedelta(seconds=delay_time), start_time + j * pd.Timedelta(seconds=delay_time)]
    # 清空变量
    mains_activ = 0
    mains_reactiv = 0
    pass
f_off.close()
f_on.close()