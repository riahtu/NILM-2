# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:23:46 2018

@author: Zejian Zhou
"""
# 一个训练的PNN网络已经完成了，现在用选出的activation来做出结果
# 这个是开始push limit，看看最多能容纳多少用电器同时使用,这个是函数版本
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import matplotlib.patches as mpatches

from nilmtk import DataSet

import numpy as np
import pandas as pd


# transfer a list of timestamp to combined timestamp
def list_to_time(l_time):
    time = l_time[0]
    l_time.pop(0)
    for i in l_time:
        time = time.append(i)
    return time


# truncate data time index
def truncate_time(l_time, up, down):
    for i in l_time:
        if i >= up or i <= down:
            l_time = l_time.drop(i)
    return l_time


# 形成针状序列，根据计算出的activate时间
def reconstruct(original, extract_index):
    temp = original.copy()
    temp[:] = 0
    for i in extract_index:
        temp[i] = original[i]
    return temp


# 定义一个函数，可以按固定方向找到最近的零点，正表示往后找，负表示往前找, 周期必须是3s
def find_close_zero(data, start_index, direction):
    # 输入：pandas series，pandas timestamp， int
    # 输出：timestamp
    # 断言输入的index不能为0，为零就没有意义了
    assert data[start_index] != 0, \
        'find close zero, data at %r is already zero' % (start_index)
    while data[start_index] != 0:
        start_index = start_index + direction * pd.Timedelta(seconds=1)
        # 数据点中有些缺失，这样跳过来增强鲁棒性。 TODO: A robuster algorithm to handle data loss
        while not data.index.contains(start_index):
            start_index = start_index + direction * pd.Timedelta(seconds=1)
    return start_index


# 定义一个函数，跳过空余的零
def skip_zeros(data, start_index, direction):
    assert data[start_index] == 0, \
        'skip zeros, data at %r is not zero' % (start_index)
    while data[start_index] == 0:
        start_index = start_index + direction * pd.Timedelta(seconds=1)
        # 数据点中有些缺失，这样跳过来增强鲁棒性。 TODO: A robuster algorithm to handle data loss
        while not data.index.contains(start_index):
            start_index = start_index + direction * pd.Timedelta(seconds=1)
    return start_index


# 自己做一个取少时间的函数，因为找不到TAT
def find_less_time(prev, central, post):
    if abs(prev - central) > abs(post - central):
        return prev
    else:
        return post

# 二阶高斯得到event
def get_activation(input_data, window1=100):
    # perform gaussian smoothing
    mains_gauss = input_data.rolling(window=window1, win_type='gaussian', center=True).mean(std=10)
    # perform derivation of gaussian smoothing
    gauss_d = mains_gauss.diff()
    # 选择函数，滤掉过小的梯度，因为没有用电器的需电量和关电量是慢慢上升的,选择过滤阈值为0.2
    gauss_d[abs(gauss_d) < 1] = 0
    # perform second derivation of gaussian smoothing to 计算极值
    gauss_dd = gauss_d.diff()
    # 提取二阶导数零点
    temp_gauss = 0
    zeros_index = []  # maxima's indexes/time stamps
    # zeros=[]
    for i in gauss_dd:
        if temp_gauss * i < 0:
            zeros_index.append(gauss_dd[gauss_dd == i].index)
        temp_gauss = i
    # 先变成timestamp
    if len(zeros_index) == 0:
        #print('no events found!')
        return zeros_index, zeros_index
    event_index = list_to_time(zeros_index)

    # 以上是得到了大概的event时间，现在开始获取准确的event时间和功率增量
    #  第二级信号恢复，小高斯平滑被用于获得精确的activation time
    # 小高斯平滑
    mains_gauss2 = input_data.rolling(window=9, win_type='gaussian', center=True).mean(std=10)
    # mains_gauss2=mains.rolling(window=9, win_type='gaussian', center=True).mean(std=10) # good
    # 求导
    gauss_d2 = mains_gauss2.diff()
    # 滤掉小刺波
    gauss_d2[abs(gauss_d2) < 2] = 0
    # 将导数值按第二高斯平滑器零点位置求出来
    activates_d = gauss_d2[event_index]

    activates_split = []
    for i in activates_d.index:
        t = i
        if gauss_d2[i] == 0:
            # 先往前找非零点
            t_prev = skip_zeros(data=gauss_d2, start_index=t, direction=-1)
            # 再往后找非零点
            t_post = skip_zeros(data=gauss_d2, start_index=t, direction=1)
            # 找一个最近的非零点
            if abs(t_prev - t) < abs(t_post - t):
                # 说明是往前找到的非零点,后点就是目前这个点
                t_post = t_prev
                # 前点就是往前找的前点
                t_prev = find_close_zero(data=gauss_d2, start_index=t_prev, direction=-1)
            else:
                # 说明是往后找到的非零点,前点就是目前这个点
                t_prev = t_post
                # 后点就是往后找的零点
                t_post = find_close_zero(data=gauss_d2, start_index=t_post, direction=1)
        else:
            # 往前后找0
            t_prev = find_close_zero(data=gauss_d2, start_index=t, direction=-1)
            t_post = find_close_zero(data=gauss_d2, start_index=t, direction=1)
        activates_split.append(t_prev)
        activates_split.append(t_post)
    # 转换时间戳
    activates_split = gauss_d2[activates_split].index
    # 求两个activates之间的数据平均数，这样再match的时候结果会更好，这就是ideal化的数据
    mains_ideal = mains_gauss2.copy()  # 拷贝一份原数据，永远不要对原数据做动作
    # 第一个不要了，成功了之后手动处理一下
    for i in range(1, activates_split.size - 1):
        # 选定好区间求平均，要不然就是胡乱求了
        if i % 2 == 1:
            mains_ideal[activates_split[i]:activates_split[i + 1]] \
                = mains_ideal[activates_split[i]:activates_split[i + 1]].mean()
    # 现在的问题就是计算ideal 数据的差值了,差值的计算方式是后值-前值，前值是偶数，后值是奇数
    # 同理，第一个不要了
    activates_diff = mains_gauss2[event_index].copy()  # 用于存放difference

    for i in range(activates_split.size - 1):
        if i % 2 == 0:
            activates_diff[activates_diff.index[int(i / 2)]] = mains_ideal[activates_split].iloc[i + 1] \
                                                               - mains_ideal[activates_split].iloc[i]
            pass

    # 获得activation on/off 的时间
    # search the activation time
    activation_start = []
    activation_end = []
    # min_index=0
    # 分别ON/OFF
    for i in range(activates_diff.size):
        if activates_diff[i] > 0:
            activation_start.append(activates_diff[[i]])
        else:
            activation_end.append(activates_diff[[i]])
    if len(activation_start) == 0:
        pass
    else:
        activation_start = list_to_time(activation_start)
    if len(activation_end) == 0:
        pass
    else:
        activation_end = list_to_time(activation_end)
    pass
    return activation_start, activation_end

# 从active power得到的 event time之后，我们需要得到这个时间的其他数据，比如无功，电流，电压之类的
def get_others(input_data, on_time, off_time):
    activation_start=[]
    activation_end=[]

    # 小高斯平滑
    mains_gauss2 = input_data.rolling(window=9, win_type='gaussian', center=True).mean(std=10)
    # mains_gauss2=mains.rolling(window=9, win_type='gaussian', center=True).mean(std=10) # good
    # 求导
    gauss_d2 = mains_gauss2.diff()
    # 滤掉小刺波
    gauss_d2[abs(gauss_d2) < 2] = 0
    # 将导数值按第二高斯平滑器零点位置求出来
    if len(on_time) != 0:
        activation_start = gauss_d2[on_time]
    if len(off_time) != 0:
        activation_end = gauss_d2[off_time]
    return activation_start, activation_end


# 画个识别出的event
def plot_event(input_data, on_event, off_event):
    # 画mains数据
    fig = plt.figure(figsize=[20, 10])
    ax = fig.add_subplot(1, 1, 1)
    input_data.plot()
    # 绘制activation 点
    for i in on_event.index:
        plt.plot(i, on_event[i], 'r*', markersize=10)
    for i in off_event.index:
        plt.plot(i, 0, 'ro', markersize=8)
    ax.legend(('Active power', 'ON', 'OFF'))
    plt.title('Events extraction example')
    plt.ylabel('Active power [W]')
    plt.xlabel('Time [sec]')
    plt.show()

# PNN training function, mainly for traning data generalization
def predict_appliance(pnn, in_active, in_reactiv, flag_off = False):
    # 准备training set 和 target set
    # on event
    training_set = np.empty([1, 2])
    # get active power data and reactive power data as a numpy
    activ = np.array([in_active.values]).T
    reactiv = np.array([in_reactiv.values]).T
    tr = np.concatenate((activ, reactiv), axis=1)
    training_set = np.concatenate((training_set, tr), axis=0)
    if flag_off:
        training_set = abs(training_set)
    # 删掉第一个空元素
    training_set = np.delete(training_set, 0, 0)
    y_predicted_on = pnn.predict(training_set)
    return y_predicted_on

# 在功率图上画已经识别出来的用电器
def plot_appliance(in_data, app_on, app_off, on_event_active, off_event_active):
    fig = plt.figure(figsize=[20, 10])
    ax = fig.add_subplot(1, 1, 1)
    in_data.plot()
    # 绘制activation 点
    for i in range(app_on.size):
        if app_on[i] == 0:
            plt.plot(on_event_active.index[i], on_event_active[i], 'r^', markersize=10)
        elif app_on[i] == 2:
            plt.plot(on_event_active.index[i], on_event_active[i], 'r*', markersize=10)
    for i in range(app_off.size):
        if app_off[i] == 0:
            plt.plot(off_event_active.index[i], off_event_active[i], 'rs', markersize=10)
        elif app_off[i] == 2:
            plt.plot(off_event_active.index[i], off_event_active[i], 'r+', markersize=10)
    ax.legend(('Active power', 'ON', 'OFF'))
    plt.title('Events extraction example')
    plt.ylabel('Active power [W]')
    plt.xlabel('Time [sec]')
    plt.show()

