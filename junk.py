import pandas as pd
from nilmtk import DataSet

import zejian_nilm2 as zz

iawe = DataSet('data/iawe.h5')
# fridge和computer的数据从7-01开始不能用了， 7-20 开始好了，后面都不能用，21号能用一天
iawe.set_window(start='6-20-2013', end='6-21-2013')
elec = iawe.buildings[1].elec
mains_activ = 0
mains_reactiv = 0
start_time = next(elec['fridge'].load(ac_type='active'))['power', 'active'].index[0]
delay_time = '0 days 00:01:00'
tspan=[start_time, start_time + pd.Timedelta(delay_time)]

#创建一个buffer
mains_buffer = pd.Series([])
mains_buffer_reactiv = pd.Series([])
buffer_size = 1 * 30 * 60  # buffer_size is one hour
buffer_ready = False

APPLIANCES = ['fridge', 'computer']

for j in range(2, 1600):
    for i in APPLIANCES:
        #print(elec[i].available_columns())
        if type(mains_activ) == type(0):
            mains_temp = mains_activ
        else:
            mains_temp = mains_activ.copy()
        mains_activ += next(elec[i].load(ac_type='active'))['power', 'active'].fillna(0)[tspan[0]:tspan[1]]
        mains_activ = mains_activ.fillna(0)
        adder = mains_temp-mains_activ
        mains_activ = mains_temp-adder

        if type(mains_reactiv) == type(0):
            mains_temp = mains_reactiv
        else:
            mains_temp = mains_reactiv.copy()
        mains_reactiv += next(elec[i].load(ac_type='reactive'))['power', 'reactive'].fillna(0)[tspan[0]:tspan[1]]
        mains_reactiv = mains_reactiv.fillna(0)
        adder = mains_temp-mains_reactiv
        mains_activ = mains_temp-adder