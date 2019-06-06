# -*- coding: utf-8 -*-
import pickle
import paho.mqtt.client as paho
import pandas as pd
import zejian_nilm4 as zz
import json


# mqtt initialization
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))


def on_message(client, userdata, msg):
    # print(msg.topic+" "+msg.payload.decode("utf-8","ignore"))
    main_algorithm(json.loads(msg.payload.decode("utf-8","ignore")))


def main_algorithm(message):
    global j, tspan, pnn, mains_buffer_P, mains_buffer_I, mains_buffer_PF, buffer_size, buffer_ready, start_time
    global events, mains, mains_P, mains_I, mains_PF, activ_on_list, activ_off_list

    # Obtain the new values
    j = j + 1
    print(j)
    # print(message)
    loc = tspan[0]
    temp_P = message['active']
    temp_I = message['current']
    temp_PF = message['pf']
    # print(temp_P)
    # 保存到buffer and remove duplicates
    mains_buffer_P.append(temp_P)
    mains_buffer_I.append(temp_I)
    mains_buffer_PF.append(temp_PF)
    # print(mains_buffer_P)

    # check to see if buffer is ready
    # if mains_buffer_P.size > buffer_size+1 or buffer_ready:
    if len(mains_buffer_P) >= buffer_size:
        buffer_ready = True
    else:
        buffer_ready = False

    # buffer 满了再检测
    if buffer_ready:
        # print("detecting...")
        series_index = range(j, j + 200)
        # event detection 和 active/ reactive power 计算
        buffer_df_P = pd.Series(mains_buffer_P,index=series_index)
        # print(buffer_df_P)
        # break
        pass
        pass
        pass
        # print('SS1')
        on_event_P, off_event_P, activates = zz.get_activation(buffer_df_P, window1=20)

        # 如果没有event就出去重新来，期间保存好这包数据当作buffer
        if len(on_event_P) == 0:
            on_in = []
            if len(off_event_P) == 0:
                # 增加时间
                tspan = [start_time + (j - 1) * 1, start_time + j * 1]
            else:
                off_in = off_event_P.index
                buffer_df_I = pd.Series(mains_buffer_I,index=series_index)
                on_event_I, off_event_I = zz.get_others(buffer_df_I, activates) # ###########first step
        else:
            on_in = on_event_P.index
            if len(off_event_P) == 0:
                off_in = []
            else:
                off_in = off_event_P.index
            buffer_df_I = pd.Series(mains_buffer_I,index=series_index)
            on_event_I, off_event_I = zz.get_others(buffer_df_I, activates)  # first step

        # check if the event sz is empty
        if len(activates) != 0:
            on_event_I, off_event_I = zz.get_others(buffer_df_I, activates)  # first step
            if len(on_event_P) != len(on_event_I):
                on_event_I = []
                on_event_P = []
            if len(off_event_P) != len(off_event_I):
                off_event_I = []
                off_event_P = []
        else:
            on_event_I = []
            off_event_I = []
            on_event_PF = []
            off_event_PF = []

        # do prediction
        if len(on_event_P) != 0:
            appliance_on = zz.predict_appliance(pnn, on_event_P, on_event_I)
            for k in range(appliance_on.size):
                # 对照以往的activation，得到现在的activation
                if on_event_P.index[k] > activ_on_list[-1]:
                    activ_on_list.append(on_event_P.index[k])
                    if appliance_on[k] == 0:
                        print(on_event_P.index[k], 'water heater on')
                    elif appliance_on[k] == 1:
                        print(on_event_P.index[k], 'TV on')
                    elif appliance_on[k] == 2:
                        print(on_event_P.index[k], 'Dryer on')
                    elif appliance_on[k] == 3:
                        print(on_event_P.index[k], 'ElectromagneticCooker on')
                    elif appliance_on[k] == 4:
                        print(on_event_P.index[k], 'Light on')
                    elif appliance_on[k] == 5:
                        print(on_event_P.index[k], 'Microwave on')
                    elif appliance_on[k] == 6:
                        print(on_event_P.index[k], 'Monitor on')
                    elif appliance_on[k] == 7:
                        print(on_event_P.index[k], 'Washer on')
        if len(off_event_P) != 0:
            appliance_off = zz.predict_appliance(pnn, off_event_P, off_event_I, flag_off=True)
            for k in range(appliance_off.size):
                # 对照以往的activation，得到现在的activation
                if off_event_P.index[k] > activ_off_list[-1]:
                    activ_off_list.append(off_event_P.index[k])
                    if appliance_off[k] == 0:
                        print(off_event_P.index[k], 'water heater off')
                    elif appliance_off[k] == 1:
                        print(off_event_P.index[k], 'TV off')
                    elif appliance_off[k] == 2:
                        print(off_event_P.index[k], 'Dryer off')
                    elif appliance_off[k] == 3:
                        print(off_event_P.index[k], 'ElectromagneticCooker off')
                    elif appliance_off[k] == 4:
                        print(off_event_P.index[k], 'Light off')
                    elif appliance_off[k] == 5:
                        print(off_event_P.index[k], 'Microwave off')
                    elif appliance_off[k] == 6:
                        print(off_event_P.index[k], 'Monitor off')
                    elif appliance_off[k] == 7:
                        print(off_event_P.index[k], 'Washer off')

        # pop buffer
        del mains_buffer_P[0]
        del mains_buffer_I[0]
        del mains_buffer_PF[0]
    # 增加时间
    tspan = [start_time + (j - 1) * 1, start_time + j * 1]


if __name__ == "__main__":
    client = paho.Client("raspberrypi")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("127.0.0.1", 1883, 60)  ## The edge broker
    # 读入神经网络 PNN
    pnn = pickle.load(open('../params/pnn.txt', "rb"))
    # 创建一个buffer
    mains_buffer_P = []
    mains_buffer_I = []
    mains_buffer_PF = []
    buffer_size = 200  # buffer_size is 200
    buffer_ready = False
    # pre-define parameters
    start_time = 0
    events, mains = {}, {}
    mains_P = 0
    mains_I = 0
    mains_PF = 0
    tspan = [start_time, start_time + 1]
    j=1
    # 保存activation num
    activ_on_list = [0]
    activ_off_list = [0]

    # Start
    client.subscribe('nilm')
    client.loop_forever()
