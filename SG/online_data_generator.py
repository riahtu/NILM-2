import pandas as pd
import paho.mqtt.client as paho
import time
import json

#mqtt initialization
client = paho.Client("laptop")
client.connect("127.0.0.1", 1883, 60)  ## The edge broker

#读入数据
water_heater = pd.read_excel ('../data/WaterHeater/WaterHeaterData.xlsx')
TV = pd.read_excel ('../data/WaterHeater/WaterHeaterData.xlsx')
data_dic={'water heater': water_heater,
          'TV': TV}

#合成数据
APPLIANCES = ['water heater']
mains_P = 0
mains_I = 0
mains_PF = 0
for i in APPLIANCES:
    mains_P += data_dic[i]['Active Power (W)']
    mains_I += data_dic[i]['Current (A)']
    mains_PF += data_dic[i]['Power factor']

#Send data
for p,i,pf in zip(mains_P,mains_I,mains_PF):
    message = {
        "time": time.time(),
        "active": p,
        "current": i,
        "pf": pf
    }
    res = client.publish("nilm", json.dumps(message))
    print("new data sent", res)
    time.sleep(0.3)