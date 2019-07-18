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
recordTime = 10 # expected record time

if __name__ == "__main__":
    while 1:
        global counter, bufferI, bufferP, bufferU
        if counter < recordTime:
            counter = counter + 1
            writer.writerow([0,1,2,3])
            print(counter)
        else:
            f.close()
            print("Data collection finished")
            exit()

