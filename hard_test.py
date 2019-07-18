# -*- coding: utf-8 -*-
#that 12 classes
import pickle
import paho.mqtt.client as paho
import pandas as pd
import zejian_nilm4 as zz
import keras
import json
from typing import Dict, Any
import numpy as np
from numpy import genfromtxt
import csv

testTime = 30*60 #second

# mqtt initialization
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))


def on_message(client, userdata, msg):
    global writer, f
    thisData = msg.payload.decode("utf-8", "ignore").split('/')
    writer.writerow(thisData)
    main_algorithm(msg.payload.decode("utf-8", "ignore"))


def on_log(client, userdata, level, buf):
    print("log: ",buf)


def main_algorithm(message):
    global j, pnn, mains_buffer_P, mains_buffer_I, mains_buffer_U, buffer_size, buffer_ready, hisDP, previousEvent, fLog, f
    # Obtain the new values
    j = j + 1
    if j >= testTime:
        f.close()
        fLog.close()
        exit()
    print(j)
    temp_U, temp_I, temp_P, _ = message.split('/')
    print(temp_U, temp_I, temp_P)
    temp_I = float(temp_I)
    temp_U = float(temp_U)
    temp_P = float(temp_P)
    # 保存到buffer and remove duplicates
    mains_buffer_P.append(float(temp_P))
    mains_buffer_I.append(float(temp_I))
    mains_buffer_U.append(float(temp_U))

    # check to see if buffer is ready
    if len(mains_buffer_P) >= buffer_size:
        buffer_ready = True
    else:
        buffer_ready = False

    # buffer 满了再检测
    if buffer_ready:
        last_event = j
        # switch to numpy
        this_buffer = np.concatenate(
            (np.array([mains_buffer_P]), np.array([mains_buffer_I]), np.array([mains_buffer_U])), axis=0)
        roughRes, _ = zz.get_events_cnn_trans_dbg(this_buffer, net=cnnDic, pnnDic=2, previousEvent=previousEvent)
        if previousEvent != 'hold...' and previousEvent != 'hold....':
            previousEvent = roughRes
        #second classfication
        detailRes, hisDP = zz.detail_class(roughClass=roughRes, thisBuffer=this_buffer, pnnDic=-1, hisDP=hisDP)
        print(detailRes)
        print(str(j)+' '+detailRes, file=fLog)
        # pop buffer
        del mains_buffer_P[0]
        del mains_buffer_I[0]
        del mains_buffer_U[0]
        pass

if __name__ == "__main__":
    client = paho.Client("pc")
    client.on_connect = on_connect
    client.on_message = on_message
    # client.on_log = on_log  # for debug only
    client.connect("192.168.1.4", 1883, 300)  # The edge broker

    # 创建一个buffer
    mains_buffer_P = []
    mains_buffer_I = []
    mains_buffer_U = []
    buffer_size = int(2*60)  # buffer_size is 200
    buffer_ready = False

    # history delta P to detect multi-state
    hisDP = 0
    flagStateChange = False
    previousEvent = 0


    # stuffed buffer
    oneFatBuffer = genfromtxt('data/stuffed_buffer.csv', delimiter=',')
    mains_buffer_P = oneFatBuffer[0:buffer_size-2, 1].tolist()
    mains_buffer_I = oneFatBuffer[0:buffer_size - 2, 1].tolist()
    mains_buffer_U = oneFatBuffer[0:buffer_size - 2, 1].tolist()

    # 保存测试结果文
    f = open("data/test_dat.csv", 'w')
    fLog = open("data/test_log.txt", 'w')
    writer = csv.writer(f)
    writer.writerow("this is a new row")
    # read in CNN
    print("CNN loading ...")
    # 读入神经网络 PNN
    cnnDic = {'class1': keras.models.load_model('LSTM/model_transclass1.h5'),
              'class5': keras.models.load_model('LSTM/model_transclass5.h5'),
              'class7': keras.models.load_model('LSTM/model_transclass7.h5')}
    print("CNN loaded!")

    last_event = 0
    j = 1

    client.subscribe('data')
    client.loop_forever()

