# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:23:46 2019
先识别一些已知的用户数据，然后再训练PNN得到网络后做预测
online feed data 为了解决这个问题
1. create a buffer to enlarge the dataset, dump the data only when buffer is full
queue style buffer
2. 读入Excel
@author: Zejian Zhou
"""
import pickle
# TODO: 别忘了调整参数
from typing import Dict, Any

import pandas as pd
from nilmtk import DataSet

import zejian_nilm3 as zz

#读入数据
water_heater = pd.read_excel ('data/Water Heater/kettle data.xlsx')
TV = pd.read_excel ('data/TV/TV_data.xlsx')
data_dic={'water heater': water_heater,
          'TV': TV}

# 做一个合成的数据
APPLIANCES = ['water heater', 'TV']
events: Dict[Any, Any] = {}
mains: Dict[Any, Any] = {}
mains_P = 0
mains_I = 0
mains_PF = 0
start_time = 1001
tspan=[start_time, start_time + 1]
# 读入神经网络 PNN
pnn = pickle.load(open('params/pnn.txt', "rb"))

#创建一个buffer
mains_buffer_P = pd.Series([])
mains_buffer_I = pd.Series([])
mains_buffer_PF = pd.Series([])
buffer_size = 200  # buffer_size is 200
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
    mains_PF += data_dic[i]['Power factor']

# 如果遇到丢掉的数据就用前后来非零来代替 TODO: fix it with a better algorithm to handle package loss
mains_P = mains_P.fillna(method='ffill')
mains_I = mains_I.fillna(method='ffill')
mains_PF = mains_PF.fillna(method='ffill')

for j in range(2, 1000):
    temp_P = pd.Series(mains_P.iloc[tspan[0]], index=[mains_P.index[tspan[0]]])
    temp_I = pd.Series(mains_I.iloc[tspan[0]], index=[mains_I.index[tspan[0]]])
    temp_PF = pd.Series(mains_PF.iloc[tspan[0]], index=[mains_PF.index[tspan[0]]])
    # 保存到buffer and remove duplicates
    mains_buffer_P = mains_buffer_P.append(temp_P)
    mains_buffer_P = mains_buffer_P[~mains_buffer_P.index.duplicated()]
    mains_buffer_I = mains_buffer_I.append(temp_I)
    mains_buffer_I = mains_buffer_I[~mains_buffer_I.index.duplicated()]
    mains_buffer_PF = mains_buffer_PF.append(temp_PF)
    mains_buffer_PF = mains_buffer_PF[~mains_buffer_PF.index.duplicated()]

    # check to see if buffer is ready
    if mains_buffer_P.size > buffer_size+1 or buffer_ready:
        mains_buffer_P = mains_buffer_P.drop(mains_buffer_P.index[0:mains_buffer_P.size-buffer_size])
        mains_buffer_I = mains_buffer_I.drop(mains_buffer_I.index[0:mains_buffer_I.size - buffer_size])
        mains_buffer_PF = mains_buffer_PF.drop(mains_buffer_PF.index[0:mains_buffer_PF.size - buffer_size])
        buffer_ready = True
    else:
        buffer_ready = False

    # buffer 满了再检测
    if buffer_ready:
        # event detection 和 active/ reactive power 计算
        on_event_P, off_event_P = zz.get_activation(mains_buffer_P, window1=20)
        # 如果没有event就出去重新来，期间保存好这包数据当作buffer
        if len(on_event_P) == 0 and len(off_event_P) == 0:
            mains_P = 0
            mains_I = 0
            print(tspan[0], tspan[1], 'no event')
            # 增加时间
            tspan = [start_time + (j-1)*1, start_time + j*1]
            continue
        if len(on_event_P) == 0:
            on_in = []
        else:
            on_in=on_event_P.index
        if len(off_event_P) == 0:
            off_in=[]
        else:
            off_in=off_event_P.index
        on_event_I, off_event_I = zz.get_others(mains_buffer_I, on_in, off_in)
        on_event_PF, off_event_PF = zz.get_others(mains_buffer_PF, on_in, off_in)

        #zz.plot_event(mains['computer active power'], events['computer active on'], events['computer active off'])

        # do prediction
        if len(on_event_P) != 0:
            appliance_on = zz.predict_appliance(pnn, on_event_P, on_event_I, on_event_PF)
        if len(off_event_P) != 0:
            appliance_off = zz.predict_appliance(pnn, off_event_P, off_event_I, off_event_PF, flag_off=True)
        if len(on_event_P) != 0:
            for k in range(appliance_on.size):
                # 对照以往的activation，得到现在的activation
                if on_event_P.index[k] > activ_on_list[-1]:
                    activ_on_list.append(on_event_P.index[k])
                    if appliance_on[k] == 0:
                        print(on_event_P.index[k], 'water heater on')
                        print(on_event_P.index[k], 'water heater on', file=f_on)
                    elif appliance_on[k] == 2:
                        print(on_event_P.index[k], 'computer on')
                        print(on_event_P.index[k], 'computer on', file=f_on)
                    elif appliance_on[k] == 1:
                        print(on_event_P.index[k], 'TV on')
                        print(on_event_P.index[k], 'TV on', file=f_on)

        if len(off_event_P) != 0:
            for k in range(appliance_off.size):
                # 对照以往的activation，得到现在的activation
                if off_event_P.index[k] > activ_off_list[-1]:
                    activ_off_list.append(off_event_P.index[k])
                    if appliance_off[k] == 0:
                        print(off_event_P.index[k], 'water heater off')
                        print(off_event_P.index[k], 'water heater off', file=f_off)
                    elif appliance_off[k] == 2:
                        print(off_event_P.index[k], 'computer off')
                        print(off_event_P.index[k], 'computer off', file=f_off)
                    elif appliance_off[k] == 1:
                        print(off_event_P.index[k], 'TV off')
                        print(off_event_P.index[k], 'TV off', file=f_off)
        # pop buffer
        mains_buffer_P = mains_buffer_P.drop(mains_buffer_P.index[0])
        mains_buffer_I = mains_buffer_I.drop(mains_buffer_I.index[0])
        mains_buffer_PF = mains_buffer_PF.drop(mains_buffer_PF.index[0])
        pass
    # 增加时间
    tspan = [start_time + (j - 1) * 1, start_time + j * 1]
f_off.close()
f_on.close()