# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 2019
get the label data of appliances time
@author: Zejian Zhou
"""
import pickle
# TODO: 别忘了调整参数
from typing import Dict, Any

import pandas as pd
from nilmtk import DataSet
import matplotlib.pyplot as plt

import zejian_nilm2 as zz

# 保存label
f_on = open('res/real_app_on.txt', 'w')
f_off = open('res/real_app_off.txt', 'w')

iawe = DataSet('data/iawe.h5')
# fridge和computer的数据从7-01开始不能用了， 7-20 开始好了，后面都不能用，21号能用一天
iawe.set_window(start='6-20-2013', end='6-21-2013')
elec = iawe.buildings[1].elec

APPLIANCES = ['fridge', 'computer']

fridge = elec['fridge']
computer = elec['computer']

computer.plot()

fridge_activ = fridge.get_activations(on_power_threshold=40)
computer_activ = computer.get_activations(on_power_threshold=20)

for i in fridge_activ:
    print(i.index[0], 'fridge on', file=f_on)
    print(i.index[-1], 'fridge off', file=f_off)
for i in computer_activ:
    print(i.index[0], 'computer on', file=f_on)
    print(i.index[-1], 'computer off', file=f_off)

plt.show()
f_off.close()
f_on.close()