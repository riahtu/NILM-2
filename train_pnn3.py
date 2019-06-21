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
from numpy import genfromtxt

# train PNN for Dryer
dryer_pnn = algorithms.PNN(std=150, verbose=False)
datS = genfromtxt('data/self_made_data/dryer_pnn.csv', delimiter=',')
training_set = datS[:, 0]
target_set = datS[:, 1]
dryer_pnn.train(training_set, target_set)
pickle.dump(dryer_pnn, open('params/dryer_pnn.txt', 'wb'))

# train PNN for cooker
cooker_pnn = algorithms.PNN(std=150, verbose=False)
datS = genfromtxt('data/self_made_data/cooker_pnn.csv', delimiter=',')
training_set = datS[:, 0]
target_set = datS[:, 1]
cooker_pnn.train(training_set, target_set)
pickle.dump(cooker_pnn, open('params/cooker_pnn.txt', 'wb'))

# train PNN for fridge
fridge_pnn = algorithms.PNN(std=150, verbose=False)
datS = genfromtxt('data/self_made_data/fridge_pnn.csv', delimiter=',')
training_set = datS[:, 0]
target_set = datS[:, 1]
fridge_pnn.train(training_set, target_set)
pickle.dump(fridge_pnn, open('params/fridge_pnn.txt', 'wb'))


# train PNN for hair dryer
hairDryer_pnn = algorithms.PNN(std=150, verbose=False)
datS = genfromtxt('data/self_made_data/hairDryer_pnn.csv', delimiter=',')
training_set = datS[:, 0]
target_set = datS[:, 1]
hairDryer_pnn.train(training_set, target_set)
pickle.dump(hairDryer_pnn, open('params/hairDryer_pnn.txt', 'wb'))

# train PNN for heat fan
heatFan_pnn = algorithms.PNN(std=150, verbose=False)
datS = genfromtxt('data/self_made_data/heatFan_pnn.csv', delimiter=',')
training_set = datS[:, 0]
target_set = datS[:, 1]
heatFan_pnn.train(training_set, target_set)
pickle.dump(heatFan_pnn, open('params/heatFan_pnn.txt', 'wb'))

# train PNN for light
light_pnn = algorithms.PNN(std=90, verbose=False)
datS = genfromtxt('data/self_made_data/light_pnn.csv', delimiter=',')
training_set = datS[:, 0]
target_set = datS[:, 1]
light_pnn.train(training_set, target_set)
pickle.dump(light_pnn, open('params/light_pnn.txt', 'wb'))

# train PNN for microwave
microwave_pnn = algorithms.PNN(std=150, verbose=False)
datS = genfromtxt('data/self_made_data/microwave_pnn.csv', delimiter=',')
training_set = datS[:, 0]
target_set = datS[:, 1]
microwave_pnn.train(training_set, target_set)
pickle.dump(light_pnn, open('params/microwave_pnn.txt', 'wb'))

# train PNN for tv
tv_pnn = algorithms.PNN(std=40, verbose=False)
datS = genfromtxt('data/self_made_data/tv_pnn.csv', delimiter=',')
training_set = datS[:, 0]
target_set = datS[:, 1]
tv_pnn.train(training_set, target_set)
pickle.dump(light_pnn, open('params/tv_pnn.txt', 'wb'))

# train PNN for microwave
washer_pnn = algorithms.PNN(std=100, verbose=False)
datS = genfromtxt('data/self_made_data/washer_pnn.csv', delimiter=',')
training_set = datS[:, 0]
target_set = datS[:, 1]
washer_pnn.train(training_set, target_set)
pickle.dump(light_pnn, open('params/washer_pnn.txt', 'wb'))

# train PNN for microwave
waterHeater_pnn = algorithms.PNN(std=100, verbose=False)
datS = genfromtxt('data/self_made_data/waterHeater_pnn.csv', delimiter=',')
training_set = datS[:, 0]
target_set = datS[:, 1]
waterHeater_pnn.train(training_set, target_set)
pickle.dump(light_pnn, open('params/waterHeater_pnn.txt', 'wb'))
