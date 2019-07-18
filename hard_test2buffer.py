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

import zejian_nilm4 as zz
import keras

#读入数据
HairDryer = pd.read_csv ('data/hairDryer_test.csv')
Monitor = pd.read_csv ('data/monitor_test.csv')
fridge = pd.read_csv ('data/fridge_own.csv')
tester = pd.read_csv('data/test_dat.csv')

data_dic={'HairDryer': HairDryer, 'Monitor': Monitor, 'Fridge': fridge, 'tester': tester}

# 做一个合成的数据
APPLIANCES = ['tester']
events: Dict[Any, Any] = {}
mains: Dict[Any, Any] = {}
mains_P = 0
mains_I = 0
mains_U = 0
start_time = 0
tspan=[start_time, start_time + 1]

previousEvent=0
#创建一个buffer
mains_buffer_P = pd.Series([])
mains_buffer_I = pd.Series([])
mains_buffer_U = pd.Series([])
buffer_size = 2*60  # buffer_size is 200
buffer_ready = False

# 保存测试结果文件
f_on = open('res/test_app_on.txt', 'w')
f_off = open('res/test_app_off.txt', 'w')

# 保存activation num
activ_on_list = [start_time]
activ_off_list = [start_time]

# 人工合成数据
for i in APPLIANCES:
    mains_P += data_dic[i].iloc[:, 2]
    mains_I += data_dic[i].iloc[:, 1]
    mains_U = data_dic[i].iloc[:, 0]

# 如果遇到丢掉的数据就用前后来非零来代替 TODO: fix it with a better algorithm to handle package loss
mains_P = mains_P.fillna(method='ffill')
mains_I = mains_I.fillna(method='ffill')
mains_U = mains_U.fillna(method='ffill')

# read in CNN
cnnDic = {'class1': keras.models.load_model('LSTM/model_transclass1.h5'),
          'class5': keras.models.load_model('LSTM/model_transclass5.h5'),
          'class7': keras.models.load_model('LSTM/model_transclass7.h5')}

last_event = 0

for j in range(2, 3000):
    temp_P = pd.Series(mains_P.iloc[tspan[0]], index=[mains_P.index[tspan[0]]])
    temp_I = pd.Series(mains_I.iloc[tspan[0]], index=[mains_I.index[tspan[0]]])
    temp_U = pd.Series(mains_U.iloc[tspan[0]], index=[mains_U.index[tspan[0]]])

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
        roughRes, _ = zz.get_events_cnn_trans_dbg(this_buffer, net=cnnDic, pnnDic=2, previousEvent=previousEvent)
        if roughRes != 'hold...' and roughRes != 'hold....':
            previousEvent = roughRes
        # second classfication
        detailRes, hisDP = zz.detail_class(roughClass=roughRes, thisBuffer=this_buffer, pnnDic=-1, hisDP=0)
        if detailRes!='no event':
            print(detailRes)

        # pop buffer
        mains_buffer_P = mains_buffer_P.drop(mains_buffer_P.index[0])
        mains_buffer_I = mains_buffer_I.drop(mains_buffer_I.index[0])
        mains_buffer_U = mains_buffer_U.drop(mains_buffer_U.index[0])
        pass
    # 增加时间
    tspan = [start_time + (j - 1) * 1, start_time + j * 1]
f_off.close()
f_on.close()