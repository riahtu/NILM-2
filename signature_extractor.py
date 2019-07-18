# -*- coding: utf-8 -*-
"""
this script can extract and calculate the features we care about for each appliance
@author: Zejian Zhou
"""

from zejian_nilm4 import Appliance
import pandas as pd

data_path = '../data/all_data2.hdf5'

all_data = pd.read_hdf(data_path)

a=0