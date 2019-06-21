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
# APPLIANCES = ['WaterHeater', 'TV', 'Dryer', 'ElectromagneticCooker', 'Light',
#               'Microwave', 'Monitor','HairDryer','HeatFan','PC','Washer','Laptop']
APPLIANCES = ['WaterHeater', 'TVOffOn', 'TVStdOn', 'TVStdOff', 'Dryer', 'Light',
              'Microwave', 'Monitor','HairDryerOffOn', 'HairDryerStdOn', 'HairDryerStdOff','HeatFanOffOn',
              'HeatFanStdOn', 'HeatFanStdOff','PC','Washer','Laptop']
# APPLIANCES = ['TV','Monitor','HairDryer','HeatFan','PC','Washer','Laptop']
events: Dict[Any, Any] = {}
mains: Dict[Any, Any] = {}
window = 25

events = {'WaterHeater P on': pd.Series(894), 'WaterHeater P off': pd.Series(-885),
      'WaterHeater V on': pd.Series(-5.79), 'WaterHeater V off': pd.Series(5.79),
      'WaterHeater I on': pd.Series(7.48), 'WaterHeater I off': pd.Series(-7.48),
'TVOffOn P on': pd.Series(100), 'TVOffOn P off': pd.Series(-100),
      'TVOffOn V on': pd.Series(-0), 'TVOffOn V off': pd.Series(0),
      'TVOffOn I on': pd.Series(0.8), 'TVOffOn I off': pd.Series(-8),
'TVStdOn P on': pd.Series(70.235), 'TVStdOn P off': pd.Series(-70.235),
      'TVStdOn V on': pd.Series(0), 'TVStdOn V off': pd.Series(0),
      'TVStdOn I on': pd.Series(0.55), 'TVStdOn I off': pd.Series(-0.55),
'TVStdOff P on': pd.Series(27), 'TVStdOff P off': pd.Series(-27),
      'TVStdOff V on': pd.Series(0), 'TVStdOff V off': pd.Series(0),
      'TVStdOff I on': pd.Series(0.21), 'TVStdOff I off': pd.Series(-0.21),
'Dryer P on': pd.Series(725), 'Dryer P off': pd.Series(-225),
      'Dryer V on': pd.Series(0), 'Dryer V off': pd.Series(0),
      'Dryer I on': pd.Series(7.03), 'Dryer I off': pd.Series(-4.12),
'ElectromagneticCooker P on': pd.Series(1213), 'ElectromagneticCooker P off1': pd.Series(-95), 'ElectromagneticCooker P off2': pd.Series(-1137),
      'ElectromagneticCooker V on': pd.Series(-5.79), 'ElectromagneticCooker V off': pd.Series(5.79),
      'ElectromagneticCooker I on': pd.Series(10), 'ElectromagneticCooker I off1': pd.Series(-1), 'ElectromagneticCooker I off2': pd.Series(-9.4),
'Light P on': pd.Series(54), 'Light P off': pd.Series(-54),
      'Light V on': pd.Series(0), 'Light V off': pd.Series(0),
      'Light I on': pd.Series(0.4), 'Light I off': pd.Series(-0.4),
'Microwave P on': pd.Series(1300), 'Microwave P off': pd.Series(-1325),
      'Microwave V on': pd.Series(-9), 'Microwave V off': pd.Series(9),
      'Microwave I on': pd.Series(12.4), 'Microwave I off': pd.Series(-12.5),
'Monitor P on': pd.Series(36), 'Monitor P off': pd.Series(-36),
      'Monitor V on': pd.Series(0), 'Monitor V off': pd.Series(0),
      'Monitor I on': pd.Series(0.4), 'Monitor I off': pd.Series(-0.4),
'HairDryerOffOn P on': pd.Series(1425), 'HairDryerOffOn P off': pd.Series(-1425),
      'HairDryerOffOn V on': pd.Series(-10), 'HairDryerOffOn V off': pd.Series(10),
      'HairDryerOffOn I on': pd.Series(12), 'HairDryerOffOn I off': pd.Series(-12),
'HairDryerStdOn P on': pd.Series(1004), 'HairDryerStdOn P off': pd.Series(-1004),
      'HairDryerStdOn V on': pd.Series(-2), 'HairDryerStdOn V off': pd.Series(2),
      'HairDryerStdOn I on': pd.Series(9), 'HairDryerStdOn I off': pd.Series(-9),
'HairDryerStdOff P on': pd.Series(411), 'HairDryerStdOff P off': pd.Series(-411),
      'HairDryerStdOff V on': pd.Series(-7), 'HairDryerStdOff V off': pd.Series(7),
      'HairDryerStdOff I on': pd.Series(3), 'HairDryerStdOff I off': pd.Series(-3),
'HeatFanOffOn P on': pd.Series(1389), 'HeatFanOffOn P off': pd.Series(-1389),
      'HeatFanOffOn V on': pd.Series(-10), 'HeatFanOffOn V off': pd.Series(10),
      'HeatFanOffOn I on': pd.Series(12), 'HeatFanOffOn I off': pd.Series(-12),
'HeatFanStdOn P on': pd.Series(399), 'HeatFanStdOn P off': pd.Series(-399),
      'HeatFanStdOn V on': pd.Series(-6), 'HeatFanStdOn V off': pd.Series(6),
      'HeatFanStdOn I on': pd.Series(3), 'HeatFanStdOn I off': pd.Series(-3),
'HeatFanStdOff P on': pd.Series(990), 'HeatFanStdOff P off': pd.Series(-990),
      'HeatFanStdOff V on': pd.Series(-3), 'HeatFanStdOff V off': pd.Series(3),
      'HeatFanStdOff I on': pd.Series(8), 'HeatFanStdOff I off': pd.Series(-8),
'PC P on': pd.Series(40), 'PC P off': pd.Series(-40),
      'PC V on': pd.Series(0), 'PC V off': pd.Series(0),
      'PC I on': pd.Series(0.26), 'PC I off': pd.Series(-0.2),
'Washer P on': pd.Series(690), 'Washer P off': pd.Series(-470),
      'Washer V on': pd.Series(-2), 'Washer V off': pd.Series(2),
      'Washer I on': pd.Series(8.8), 'Washer I off': pd.Series(-8),
'Laptop P on': pd.Series(50), 'Laptop P off': pd.Series(-40),
      'Laptop V on': pd.Series(-0), 'Laptop V off': pd.Series(0),
      'Laptop I on': pd.Series(0.3), 'Laptop I off': pd.Series(-0.3)}



# zz.plot_event(mains['Dryer P'], events['Dryer P on'], events['Dryer P off'])


# 训练神经网络 PNN
pnn = algorithms.PNN(std=150, verbose=False)
# 准备training set 和 target set
count = 0
training_set = np.empty([1, 2])
target_set = np.empty([1, 1])
for i in APPLIANCES:
    # get active power data and reactive power data as a numpy
    P = np.array([events[i+' P on'].values]).T
    I = np.array([events[i+' I on'].values]).T
    V = np.array([events[i + ' V on'].values]).T
    tr = np.concatenate((P, I), axis=1)
    training_set = np.concatenate((training_set, tr), axis=0)
    targ = np.empty(tr.shape[0])
    targ[:] = count
    target_set = np.append(target_set, np.array([targ]).T, axis=0)
    count += 1
for i in APPLIANCES:
    # get active power data and reactive power data as a numpy
    P = np.array([events[i+' P off'].values]).T
    I = np.array([events[i+' I off'].values]).T
    # V = np.array([events[i + ' V off'].values]).T
    tr = np.concatenate((P, I), axis=1)
    training_set = np.concatenate((training_set, tr), axis=0)
    targ = np.empty(tr.shape[0])
    targ[:] = count
    target_set = np.append(target_set, np.array([targ]).T, axis=0)
    count += 1


P = np.array([events['ElectromagneticCooker P on'].values]).T
I = np.array([events['ElectromagneticCooker I on'].values]).T
tr = np.concatenate((P, I), axis=1)
training_set = np.concatenate((training_set, tr), axis=0)
targ = np.empty(tr.shape[0])
targ[:] = count
target_set = np.append(target_set, np.array([targ]).T, axis=0)
count += 1

P = np.array([events['ElectromagneticCooker P off1'].values]).T
I = np.array([events['ElectromagneticCooker I off1'].values]).T
# V = np.array([events['ElectromagneticCooker V off1'].values]).T
tr = np.concatenate((P, I), axis=1)
training_set = np.concatenate((training_set, tr), axis=0)
targ = np.empty(tr.shape[0])
targ[:] = count
target_set = np.append(target_set, np.array([targ]).T, axis=0)
count += 1

P = np.array([events['ElectromagneticCooker P off2'].values]).T
I = np.array([events['ElectromagneticCooker I off2'].values]).T
# V = np.array([events['ElectromagneticCooker V off2'].values]).T
tr = np.concatenate((P, I), axis=1)
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