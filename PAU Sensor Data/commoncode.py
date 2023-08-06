# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 14:16:12 2023

@author: Tanisha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read dataset
datar = pd.read_csv('datac.csv', index_col = 'created_at', parse_dates = True)

datar

data = datar.iloc[:, :-5]

data

column_names=["entry_id","pH","Relative_Humidity","Temperature","EC","N","P","K","other","pH","Relative_Humidity","Temperature","EC","N","P","K"]
data.columns = column_names

data=data.iloc[:,:-8]

data=data['P']
data=pd.DataFrame(data)
data= pd.to_numeric(data['P'], errors='coerce')
data=pd.DataFrame(data)
data['P']=data['P'].mask(data['P']>=200,data.mode())

#type(data['P'])
data= pd.to_numeric(data['P'], errors='coerce')
#data['2022-08-5T8:02:0+05:30':'2022-08-5T8:0:2+05:30']=20


#data2.plot()

#resampling
data4=data.resample('T').mean()
data4=pd.DataFrame(data4)
#data4=data4['P'].interpolate(method='linear')
data4 = data4.fillna(method='ffill')
data4.plot()
data4= pd.to_numeric(data4['P'], errors='coerce')
data4.fillna(data4.mean())
data4.isna().sum()
data4=data4['2022-08-16 00:00:00+05:30':]
data4=data4[:'2023-05-09 23:00:00+05:30']
data4.to_csv('preprocessed_p.csv', index=True)
