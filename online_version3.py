# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:23:46 2019
先识别一些已知的用户数据，然后再训练PNN得到网络后做预测
online feed data 为了解决这个问题
1. create a buffer to enlarge the dataset, dump the data only when buffer is full
queue style buffer
2. 读入Excel
3. available data: Dryer, ElectromagneticCooker, HairDryer, HeatFan, Laptop, Light, MicroWave, Monitor, PC, Refrigerator,
TV, Washer, WaterHeater
@author: Zejian Zhou
"""

import pickle
# TODO: 别忘了调整参数
from typing import Dict, Any
import numpy as np
import pandas as pd
from nilmtk import DataSet

import zejian_nilm4_special as zz
import keras

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
test = pd.read_excel('data/test3.xlsx')

data_dic={'WaterHeater': WaterHeater, 'TV': TV, 'Dryer': Dryer, 'ElectromagneticCooker': ElectromagneticCooker,
          'Fridge': Fridge, 'HairDryer': HairDryer, 'HeatFan': HeatFan, 'Laptop': Laptop, 'Light': Light,
          'Microwave': Microwave, 'PC': PC, 'Monitor': Monitor, 'Washer': Washer, 'test': test}

# 做一个合成的数据
APPLIANCES = ['test']
events: Dict[Any, Any] = {}
mains: Dict[Any, Any] = {}
mains_P = 0
mains_I = 0
mains_U = 0
start_time = 0
tspan=[start_time, start_time + 1]
# 读入神经网络 PNN
pnnDic = {'pnnDryer': pickle.load(open('params/dryer_pnn.txt', "rb")),
          'pnnCooker': pickle.load(open('params/cooker_pnn.txt', "rb")),
          'pnnFridge': pickle.load(open('params/fridge_pnn.txt', "rb")),
          'pnnHairDryer': pickle.load(open('params/hairDryer_pnn.txt', "rb")),
          'pnnHeatFan': pickle.load(open('params/heatFan_pnn.txt', "rb")),
          'pnnLight': pickle.load(open('params/light_pnn.txt', "rb")),
          'pnnMicrowave': pickle.load(open('params/microwave_pnn.txt', "rb")),
          'pnnTV': pickle.load(open('params/tv_pnn.txt', "rb")),
          'pnnWasher': pickle.load(open('params/washer_pnn.txt', "rb")),
          'pnnWaterHeater': pickle.load(open('params/waterHeater_pnn.txt', "rb"))}

#创建一个buffer
mains_buffer_P = pd.Series([])
mains_buffer_I = pd.Series([])
mains_buffer_U = pd.Series([])
buffer_size = 50  # buffer_size is 200
buffer_ready = False

# 保存测试结果文件
f_on = open('res/test_app_on.txt', 'w')
f_off = open('res/test_app_off.txt', 'w')

# 保存activation num
activ_on_list = [start_time]
activ_off_list = [start_time]

# 人工合成数据
for i in APPLIANCES:
    mains_P += data_dic[i]['Active Power (W)']
    mains_I += data_dic[i]['Current (A)']
    mains_U = data_dic[i]['Voltage (v)']

# 如果遇到丢掉的数据就用前后来非零来代替 TODO: fix it with a better algorithm to handle package loss
mains_P = mains_P.fillna(method='ffill')
mains_I = mains_I.fillna(method='ffill')
mains_U = mains_U.fillna(method='ffill')

# read in CNN
net = keras.models.load_model('LSTM\model.h5')  # TODO：这个可以不用每次都读，这样效率不高

last_event = 0

for j in range(2, 3000):
    temp_P = pd.Series(mains_P.iloc[tspan[0]], index=[mains_P.index[tspan[0]]])
    temp_I = pd.Series(mains_I.iloc[tspan[0]], index=[mains_I.index[tspan[0]]])
    temp_U = pd.Series(mains_U.iloc[tspan[0]], index=[mains_U.index[tspan[0]]])
    # if temp_P.values > 550:
    #     temp_P -= 550 + 20
    #     temp_U += 6 - 0.4
    #     if temp_I.values > 5.3:
    #         temp_I -= 5.3+0.4
    # # elif temp_P.values < 0.1:
    # temp_P += 0.01
    # # if temp_I.values < 0.1:
    # temp_I += 0.01

    # if temp_P.values > 550:
    #     temp_P += 20
    #     temp_U -= 0.4
    #     if temp_I.values > 5.3:
    #         temp_I += 0.4
    # # elif temp_P.values < 0.1:
    # temp_P += 0.01
    # # if temp_I.values < 0.1:
    # temp_I += 0.01



    # 保存到buffer and remove duplicates
    mains_buffer_P = mains_buffer_P.append(temp_P)
    mains_buffer_P = mains_buffer_P[~mains_buffer_P.index.duplicated()]
    mains_buffer_I = mains_buffer_I.append(temp_I)
    mains_buffer_I = mains_buffer_I[~mains_buffer_I.index.duplicated()]
    mains_buffer_U = mains_buffer_U.append(temp_U)
    mains_buffer_U = mains_buffer_U[~mains_buffer_U.index.duplicated()]

    # check to see if buffer is ready
    if mains_buffer_P.size > buffer_size+1 or buffer_ready:
        mains_buffer_P = mains_buffer_P.drop(mains_buffer_P.index[0:mains_buffer_P.size-buffer_size])
        mains_buffer_I = mains_buffer_I.drop(mains_buffer_I.index[0:mains_buffer_I.size - buffer_size])
        mains_buffer_U = mains_buffer_U.drop(mains_buffer_U.index[0:mains_buffer_U.size - buffer_size])
        buffer_ready = True
    else:
        buffer_ready = False

    # buffer 满了再检测
    if buffer_ready:
        last_event = j
        # switch to numpy
        this_buffer = np.concatenate((np.array([mains_buffer_P]), np.array([mains_buffer_I]), np.array([mains_buffer_U])), axis=0)
        res = zz.get_events_cnn(this_buffer, net=net, pnnDic=pnnDic)
        applianceNum = res[0]
        prob = res[1]
        event = res[2]
        if applianceNum == -1:
            pass
        elif applianceNum == 0:
            if event == 0:
                print('time=', j, ' confidence=', prob, ' Dryer on', file=f_on)
            elif event == 1:
                print('time=', j, ' confidence=', prob, ' Dryer power state up', file=f_on)
            elif event == 2:
                print('time=', j, ' confidence=', prob, ' Dryer off', file=f_on)
            elif event == 3:
                print('time=', j, ' confidence=', prob, ' Dryer power state down', file=f_on)

        elif applianceNum == 1:
            if event == 0:
                print('time=', j, ' confidence=', prob, ' Cooker on', file=f_on)
            elif event == 1:
                print('time=', j, ' confidence=', prob, ' Cooker off 1', file=f_on)
            elif event == 2:
                print('time=', j, ' confidence=', prob, ' Cooker off 2', file=f_on)

        elif applianceNum == 2:
            if event == 0:
                print('time=', j, ' confidence=', prob, ' Fridge on', file=f_on)
            elif event == 1:
                print('time=', j, ' confidence=', prob, ' Fridge off', file=f_on)
            elif event == 2:
                print('time=', j, ' confidence=', prob, ' Fridge compressor off', file=f_on)

        elif applianceNum == 3:
            if event == 0:
                print('time=', j, ' confidence=', prob, ' Hair Dryer off to high', file=f_on)
            elif event == 1:
                print('time=', j, ' confidence=', prob, ' Hair Dryer low to high', file=f_on)
            elif event == 2:
                print('time=', j, ' confidence=', prob, ' Hair Dryer off to low', file=f_on)
            elif event == 3:
                print('time=', j, ' confidence=', prob, ' Hair Dryer low to off', file=f_on)
            elif event == 4:
                print('time=', j, ' confidence=', prob, ' Hair Dryer high to low', file=f_on)
            elif event == 5:
                print('time=', j, ' confidence=', prob, ' Hair Dryer high to off', file=f_on)

        elif applianceNum == 4:
            if event == 0:
                print('time=', j, ' confidence=', prob, ' Heat Fan off to high', file=f_on)
            elif event == 1:
                print('time=', j, ' confidence=', prob, ' Heat Fan high to low', file=f_on)
            elif event == 2:
                print('time=', j, ' confidence=', prob, ' Heat Fan off to low', file=f_on)
            elif event == 3:
                print('time=', j, ' confidence=', prob, ' Heat Fan low to off', file=f_on)
            elif event == 4:
                print('time=', j, ' confidence=', prob, ' Heat Fan high to low', file=f_on)
            elif event == 5:
                print('time=', j, ' confidence=', prob, ' Heat Fan high to off', file=f_on)

        elif applianceNum == 5:
            if event == 0:
                print('time=', j, ' confidence=', prob, ' Laptop on', file=f_on)
            elif event == 1:
                print('time=', j, ' confidence=', prob, ' Laptop off', file=f_on)

        elif applianceNum == 6:
            if event == 0:
                print('time=', j, ' confidence=', prob, ' Light on', file=f_on)
            elif event == 1:
                print('time=', j, ' confidence=', prob, ' Light off', file=f_on)

        elif applianceNum == 7:
            if event == 0:
                print('time=', j, ' confidence=', prob, ' Microwave on', file=f_on)
            elif event == 1:
                print('time=', j, ' confidence=', prob, ' Microwave off', file=f_on)

        elif applianceNum == 9:
            if event == 0:
                print('time=', j, ' confidence=', prob, ' PC on', file=f_on)
            elif event == 1:
                print('time=', j, ' confidence=', prob, ' PC off', file=f_on)

        elif applianceNum == 10:
            if event == 0:
                print('time=', j, ' confidence=', prob, ' TV on', file=f_on)
            elif event == 1:
                print('time=', j, ' confidence=', prob, ' TV off', file=f_on)

        elif applianceNum == 11:
            if event == 0:
                print('time=', j, ' confidence=', prob, ' Washer on', file=f_on)
            elif event == 1:
                print('time=', j, ' confidence=', prob, ' Washer off', file=f_on)

        elif applianceNum == 12:
            if event == 0:
                print('time=', j, ' confidence=', prob, ' Water Heater on', file=f_on)
            elif event == 1:
                print('time=', j, ' confidence=', prob, ' Water Heater off', file=f_on)

        # pop buffer
        mains_buffer_P = mains_buffer_P.drop(mains_buffer_P.index[0])
        mains_buffer_I = mains_buffer_I.drop(mains_buffer_I.index[0])
        mains_buffer_U = mains_buffer_U.drop(mains_buffer_U.index[0])
        pass
    # 增加时间
    tspan = [start_time + (j - 1) * 1, start_time + j * 1]
f_off.close()
f_on.close()