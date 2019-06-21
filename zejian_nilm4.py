# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:23:46 2019
这个是excel版本的库，没有时间处理的概念

@author: Zejian Zhou
"""
# 一个训练的PNN网络已经完成了，现在用选出的activation来做出结果
# 这个是开始push limit，看看最多能容纳多少用电器同时使用,这个是函数版本
import matplotlib.pyplot as plt
import pickle
import keras

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
    original_start_index=start_index
    while data[start_index] != 0:
        start_index = start_index + direction * 1
        # 数据点中有些缺失，这样跳过来增强鲁棒性。 TODO: A robuster algorithm to handle data loss
        while not data.index.contains(start_index):
            start_index = start_index + direction * 1
            # bug fix, 超出边界了以后就用原来的
            if direction > 0 and start_index > data.index[-1]:
                return original_start_index
            elif direction < 0 and start_index < data.index[0]:
                return original_start_index
    return start_index


# 定义一个函数，跳过空余的零
def skip_zeros(data, start_index, direction):
    assert data[start_index] == 0, \
        'skip zeros, data at %r is not zero' % (start_index)
    original_start_index = start_index
    while data[start_index] == 0:
        start_index = start_index + direction * 1
        # 数据点中有些缺失，这样跳过来增强鲁棒性。 TODO: A robuster algorithm to handle data loss
        while not data.index.contains(start_index):
            start_index = start_index + direction * 1
            if direction > 0 and start_index > data.index[-1]:
                return original_start_index
            elif direction < 0 and start_index < data.index[0]:
                return original_start_index
    return start_index


# 自己做一个取少时间的函数，因为找不到TAT
def find_less_time(prev, central, post):
    if abs(prev - central) > abs(post - central):
        return prev
    else:
        return post

#CNN get event
def get_activation_cnn(Xtest, model):
    predictions = model.predict(Xtest)
    a=0



# 二阶高斯得到event
def get_activation(input_P, input_I, input_V, window1=3):
    # perform gaussian smoothing
    mains_gauss = input_P.rolling(window=window1, win_type='gaussian', center=True).mean(std=10)
    # mains_gauss = input_P
    # perform derivation of gaussian smoothing
    gauss_d = mains_gauss.diff()
    # 选择函数，滤掉过小的梯度，因为没有用电器的需电量和关电量是慢慢上升的,选择过滤阈值为0.2
    gauss_d[abs(gauss_d) < 1] = 0
    eventIndex = abs(gauss_d).idxmax()
    Dp = gauss_d.loc[eventIndex]

    postiveDp = gauss_d.max()
    negativeDp = gauss_d.max()
    # TODO: take care of fridge & washer, take care of laptop and PC
    if abs(Dp) > 20 and abs(gauss_d.dropna().iloc[-4:-1].sum()+gauss_d.dropna().iloc[-1]) < 10:
        # 往后找过零点
        i = eventIndex
        while i < input_P.index[-1]:
            if gauss_d[i]*Dp <= 0:
                break
            i += 1
        # find the sign change point before event point
        j = eventIndex
        while j > input_P.index[0]:
            if gauss_d[j]*Dp <= 0:
                break
            j -= 1
        diffP = mains_gauss[i]-mains_gauss[j]
        diffI = input_I[i]-input_I[j]
        diffV = input_V[i] - input_V[j]
        if abs(diffP)<40:
            a=0
        if abs(mains_gauss.sum())>0.5*abs(diffP):
            return pd.Series(diffP, index=[eventIndex]), pd.Series(diffI, index=[eventIndex]),\
                   pd.Series(diffV, index=[eventIndex])
        else:
            return pd.Series(), pd.Series(), pd.Series()
    else:
        return pd.Series(), pd.Series(), pd.Series()




# 从active power得到的 event time之后，我们需要得到这个时间的其他数据，比如无功，电流，电压之类的
def get_others(input_P, event_index):
    activation_start=[]
    activation_end=[]

    # 小高斯平滑
    mains_gauss2 = input_P.rolling(window=20, win_type='gaussian', center=True).mean(std=10)
    # mains_gauss2 = input_P
    # mains_gauss2=mains.rolling(window=9, win_type='gaussian', center=True).mean(std=10) # good
    # 求导
    gauss_d2 = mains_gauss2.diff()
    # 滤掉小刺波
    gauss_d2[abs(gauss_d2) < 0.02] = 0
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
        temp = pd.Series(activates_diff.iloc[i], index=[activates_diff.index[i]])
        if activates_diff.iloc[i] > 0:

            activation_start.append(temp)
        else:
            activation_end.append(temp)
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


# 画个识别出的event
def plot_event(input_P, on_event, off_event):
    # 画mains数据
    fig = plt.figure(figsize=[20, 10])
    ax = fig.add_subplot(1, 1, 1)
    input_P.plot()
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
    # tr = np.concatenate((activ, reactiv), axis=1)
    # training_set = np.concatenate((training_set, tr), axis=0)
    training_set = activ
    if flag_off:
        training_set = abs(training_set)
    # 删掉第一个空元素
    # training_set = np.delete(training_set, 0, 0)
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


# label中去掉差距太大的
def purge_wrong_label(in_data, activates):
    temp_ind = []
    temp_ind2 = []
    temp_lp = in_data.median()
    for j in range(in_data.size):
        if abs(in_data.iloc[j]-temp_lp) > abs(0.1*temp_lp):
            temp_ind.append(j)
    for j in temp_ind:
        temp_ind2.append(in_data.index[j])
    for j in temp_ind2:
        in_data = in_data.drop(j)
        activates = activates.drop(j)
    return in_data, activates


# 调整尺寸方便训练模型, 有两种方式，加或者减，现在先用加
def resz_traning_set(event_P, event_I):
    std_len = max(len(event_P), len(event_I))
    if len(event_P) < std_len:
        temp = len(event_P)
        for j in range(std_len - temp):
            event_P = event_P.append(pd.Series([event_P.median()]))
    if len(event_I) < std_len:
        temp = len(event_I)
        for j in range(std_len - temp):
            event_I = event_I.append(pd.Series([event_I.median()]))
    return event_P, event_I


# the packaged
# input : t, P, I
# output : t, 种类, ON/OFF, confidence
def get_events(thisBuffer):
    # 读入神经网络 PNN
    pnn = pickle.load(open('params/pnn.txt', "rb")) # TODO：这个可以不用每次都读，这样效率不高
    events = np.full((4, 50), None)
    # 把buffer包装成series
    indexes = thisBuffer[0, :]
    mains_buffer_P = pd.Series(thisBuffer[1, :], index=indexes)
    mains_buffer_I = pd.Series(thisBuffer[2, :], index=indexes)
    on_event_P, off_event_P, activates = get_activation(mains_buffer_P, window1=20)
    # check if the event sz is empty
    if len(activates) != 0:
        on_event_I, off_event_I = get_others(mains_buffer_I, activates)
        if len(on_event_P) != len(on_event_I):
            on_event_I = []
            on_event_P = []
        if len(off_event_P) != len(off_event_I):
            off_event_I = []
            off_event_P = []
    else:
        on_event_I = []
        off_event_I = []
    # do prediction
    appliance_on = []
    appliance_off = []
    if len(on_event_P) != 0:
        appliance_on = predict_appliance(pnn, on_event_P, on_event_I)
    if len(off_event_P) != 0:
        appliance_off = predict_appliance(pnn, off_event_P, off_event_I, flag_off=True)
    # 装入输出数组
    if len(on_event_P) != 0 and len(off_event_P) != 0:
        events[0, :] = np.pad(np.concatenate((np.array([on_event_P.index]), np.array([off_event_P.index])), axis=None),
                              (0, 50 - (on_event_P.size+off_event_P.size)), 'constant', constant_values=(None, None))
        events[1, :] = np.pad(np.concatenate((appliance_on, appliance_off), axis=None), (0, 50-(on_event_P.size+off_event_P.size)),
                              'constant', constant_values=(None, None))
        events[2, :] = np.pad(np.concatenate((np.ones(appliance_on.size), np.ones(appliance_off.size)), axis=None),
                              (0, 50-(on_event_P.size+off_event_P.size)), 'constant', constant_values=(None, None))
    return events, appliance_on, appliance_off, on_event_P, off_event_P


# the packaged
# input : t, P, I
# output : t, 种类, ON/OFF, confidence
def get_events_cnn(thisBuffer, net, pnnDic):
    # repack the buffer and remove 0 s
    thisBuffer = np.maximum(thisBuffer, 0.01)
    mains_buffer_P = thisBuffer[0, :].copy()
    mains_buffer_I = thisBuffer[1, :].copy()
    mains_buffer_U = thisBuffer[2, :].copy()
    mains_buffer_P_norm = mains_buffer_P / np.max(thisBuffer[0, :])
    mains_buffer_I_norm = mains_buffer_I / np.max(thisBuffer[1, :])
    mains_buffer_U_norm = mains_buffer_U / np.max(thisBuffer[2, :])

    mains_buffer_S = np.multiply(mains_buffer_U, mains_buffer_I)
    mains_buffer_PF = np.minimum(np.divide(mains_buffer_P, mains_buffer_S), 1)

    maxInd = np.argmax(mains_buffer_P)
    minInd = np.argmin(mains_buffer_P)
    deltaP = (max(mains_buffer_P) - min(mains_buffer_P)) * np.sign(maxInd - minInd)
    mains_buffer_DP = np.full(mains_buffer_P.shape, abs(deltaP))

    maxIndI = np.argmax(mains_buffer_I)
    minIndI = np.argmin(mains_buffer_I)
    deltaI = (max(mains_buffer_I) - min(mains_buffer_I)) * np.sign(maxIndI - minIndI)
    mains_buffer_DI = np.full(mains_buffer_I.shape, abs(deltaI))


    allData = np.concatenate(([mains_buffer_P_norm], [mains_buffer_I_norm], [mains_buffer_DP/1500], [mains_buffer_PF],
                              [mains_buffer_DI/12], [mains_buffer_U_norm]),axis=0)
    allData = allData.reshape(1, allData.shape[0], allData.shape[1])
    # if abs(max(mains_buffer_P) - min(mains_buffer_I)) <= 100 or (np.var(mains_buffer_P[0:5])<100 and np.var(mains_buffer_P[-6:]) < 100):
    if abs(deltaP) <= 50:
        #no events
        return np.array([-1, -1, -1])
    else:
        #有一个缺陷是50s频率
    # classify the appliance
        predictions = net.predict(allData)
        applianceNum = np.argmax(predictions)
        probCNN = predictions[0, applianceNum]
        event = []
        eventInd = -1
        if applianceNum == 0:
            pnn = pnnDic['pnnDryer']
            if abs(deltaP)>200 and abs(deltaP)<210:
                pnn = pnnDic['pnnLight']
                event = pnn.predict_proba([deltaP])
                eventInd = np.argmax(event)
                applianceNum = 6 #special case for light
            else:
                event = pnn.predict_proba([deltaP])
                eventInd = np.argmax(event)

        elif applianceNum == 1:
            pnn = pnnDic['pnnCooker']
            event = pnn.predict_proba([deltaP])
            eventInd = np.argmax(event)

        elif applianceNum == 2:
            pnn = pnnDic['pnnFridge']
            event = pnn.predict_proba([deltaP])
            eventInd = np.argmax(event)

        elif applianceNum == 3:
            pnn = pnnDic['pnnHairDryer']
            event = pnn.predict_proba([deltaP])
            eventInd = np.argmax(event)

        elif applianceNum == 4:
            pnn = pnnDic['pnnHeatFan']
            event = pnn.predict_proba([deltaP])
            eventInd = np.argmax(event)

        elif applianceNum == 5: #laptop
            if np.var(mains_buffer_P[-5:]) < 10:
                return np.array([applianceNum, -1, 1])
            if np.var(mains_buffer_P[0:5]) < 10:
                return np.array([applianceNum, -1, 0])
            else:
                return np.array([applianceNum, -1, -1])

        elif applianceNum == 6: #light
            pnn = pnnDic['pnnLight']
            event = pnn.predict_proba([deltaP])
            eventInd = np.argmax(event)

        elif applianceNum == 7: #microwave
            pnn = pnnDic['pnnMicrowave']
            event = pnn.predict_proba([deltaP])
            eventInd = np.argmax(event)

        elif applianceNum == 9:#PC get rid of monitor
            if np.var(mains_buffer_P[-5:]) < 10:
                return np.array([applianceNum, -1, 1])
            if np.var(mains_buffer_P[0:5]) < 10:
                return np.array([applianceNum, -1, 0])
            else:
                return np.array([applianceNum, -1, -1])

        elif applianceNum == 10:  # TV
            pnn = pnnDic['pnnTV']
            event = pnn.predict_proba([deltaP])
            eventInd = np.argmax(event)

        elif applianceNum == 11:  # washer
            pnn = pnnDic['pnnWasher']
            event = pnn.predict_proba([deltaP])
            eventInd = np.argmax(event)

        elif applianceNum == 12:  # waterheater
            # pnn = pnnDic['pnnWaterHeater']
            # event = pnn.predict_proba([deltaP])
            # eventInd = np.argmax(event)
            if deltaP >0:
                return np.array([applianceNum, 1, 0])
            else:
                return np.array([applianceNum, 1, 1])
        return np.array([applianceNum, event[0, eventInd]*probCNN, eventInd])
