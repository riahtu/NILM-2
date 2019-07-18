# -*- coding: utf-8 -*-
import pickle
import paho.mqtt.client as paho
import pandas as pd
import json
from typing import Dict, Any
import numpy as np
import csv



bufferU = []
bufferP = []
bufferI = []

#open store file
f = open("data/laptop.csv", 'a')

writer = csv.writer(f)

counter = 0
recordTime = 30*60 # expected record time

# mqtt initialization
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))


def on_message(client, userdata, msg):
    global counter, bufferI, bufferP, bufferU
    if counter < recordTime:
        counter = counter+1
        thisData = msg.payload.decode("utf-8", "ignore").split('/')
        # print(thisData)
        writer.writerow(thisData)
        print(counter)
    else:
        f.close()
        print("Data collection finished")
        exit()



def on_log(client, userdata, level, buf):
    print("log: ",buf)


if __name__ == "__main__":
    client = paho.Client("raspberrypi")
    client.on_connect = on_connect
    client.on_message = on_message
    # client.on_log = on_log  # for debug only
    client.connect("192.168.1.4", 1883, 60)  ## The edge broker
    client.subscribe('data')
    client.loop_forever()

