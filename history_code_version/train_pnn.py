# -*- coding: utf-8 -*-
"""
Created on Mon May 20 2019
先识别一些已知的用户数据，然后再训练PNN得到网络后做预测
数据源为SG
@author: Zejian Zhou
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
import zejian_nilm4 as zz
import pickle
from neupy import algorithms

#读入数据
WaterHeater = pd.read_excel ('data/WaterHeater/WaterHeaterData.xlsx')
TV = pd.read_excel ('data/TV/TVData.xlsx')
Dryer = pd.read_excel('data/Dryer/DryerData.xlsx')
ElectromagneticCooker = pd.read_excel('data/ElectromagneticCooker/ElectromagneticCookerData.xlsx')
Fridge = pd.read_excel('data/Fridge/FridgeData.xlsx')
HairDryer = pd.read_excel ('data/HairDryer/HairDryerData.xlsx')
HeatFan = pd.read_excel ('data/HeatFan/HeatFanData.xlsx')
Laptop = pd.read_excel('data/Laptop/LaptopData.xlsx')
Light = pd.read_excel('data/Light/LightData.xlsx')
Microwave = pd.read_excel('data/Microwave/MicrowaveData.xlsx')
PC = pd.read_excel('data/PC/PCData.xlsx')
Monitor = pd.read_excel ('data/Monitor/MonitorData.xlsx')
Washer = pd.read_excel ('data/Washer/WasherData.xlsx')

data_dic={'WaterHeater': WaterHeater, 'TV': TV, 'Dryer': Dryer, 'ElectromagneticCooker': ElectromagneticCooker,
          'Fridge': Fridge, 'HairDryer': HairDryer, 'HeatFan': HeatFan, 'Laptop': Laptop, 'Light': Light,
          'Microwave': Microwave, 'PC': PC, 'Monitor': Monitor, 'Washer': Washer}

# event detection, some tricky devices: Hair dryer, Heat fan, Laptop, PC
APPLIANCES = ['WaterHeater', 'TV', 'Dryer', 'ElectromagneticCooker', 'Light',
              'Microwave', 'Monitor','HairDryer','HeatFan','PC','Washer','Laptop']
# APPLIANCES = ['TV','Monitor','HairDryer','HeatFan','PC','Washer','Laptop']
events: Dict[Any, Any] = {}
mains: Dict[Any, Any] = {}
window = 25
for i in APPLIANCES:
    #分解数据
    mains_P = data_dic[i]['Active Power (W)']
    mains_I = data_dic[i]['Current (A)']
    mains_PF = data_dic[i]['Power factor']
    mains_V = data_dic[i]['Voltage (v)']

    # sliding window training
    on_event_P = pd.Series()
    on_event_I = pd.Series()
    on_event_V = pd.Series()
    off_event_P = pd.Series()
    off_event_V = pd.Series()
    off_event_I = pd.Series()
    event_P = pd.Series()
    currentIndex = 0
    while currentIndex < len(mains_P)-window:
        event_P, event_I, event_V = zz.get_activation(mains_P.iloc[currentIndex:currentIndex+window],
                                    mains_I.iloc[currentIndex:currentIndex+window],
                                    mains_V.iloc[currentIndex:currentIndex+window])
        if len(event_P) != 0:
            if event_P.values > 0:
                on_event_P = on_event_P.append(event_P)
                on_event_V = on_event_V.append(event_V)
                on_event_I = on_event_I.append(event_I)
            if event_P.values < 0:
                off_event_P = off_event_P.append(event_P)
                off_event_V = off_event_V.append(event_V)
                off_event_I = off_event_I.append(event_I)
            currentIndex = currentIndex + window
        else:
            currentIndex += 1

    df = {i + ' P on': on_event_P, i + ' P off': off_event_P,
          i + ' V on': on_event_V, i + ' V off': off_event_V,
          i + ' I on': on_event_I, i + ' I off': off_event_I}
    df2 = {i + ' P': mains_P, i + ' I': mains_I, i + ' PF': mains_PF}
    pass
    events.update(df)
    mains.update(df2)

# zz.plot_event(mains['Dryer P'], events['Dryer P on'], events['Dryer P off'])


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
    # tr = np.concatenate((P, I), axis=1)
    tr = P
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