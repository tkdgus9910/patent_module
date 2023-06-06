# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:20:26 2023

@author: tmlab
"""

import pandas as pd

directory = 'D:/SNU/TILAB - 문서/DB/patent/wisdomain/ev_hev_battery/'
data = pd.read_csv(directory + 'CSV2211203050.csv', skiprows = 4)
# data
data = data.loc[data['출원일']]