# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:09:47 2023

@author: Tanisha
"""


# Import Basic Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df5 = pd.read_excel('Sensor Data-October 02 2022.xlsx', index_col = 'created_at', parse_dates = True)

df5


column_names=["entry_id","pH","Relative_Humidity","Temperature","EC","N","P","K","other"]
df5.columns = column_names


df5

df5 = df5.iloc[1: , :]

df5

df5 = df5.iloc[:, :-1]

df5

df5=df5['N']
type(df5.index)


df5.index.asfreq = 'S'

df5 = pd.DataFrame(df5)

df5
df5= pd.to_numeric(df5['N'], errors='coerce')
df5.plot()

df5['N'] = df5['N'].ffill()
df5= pd.to_numeric(df5['N'], errors='coerce')

df5

df5=df5.resample(rule='H').mean()

#df5.dropna(inplace = True)

df5
df5 = pd.DataFrame(df5)
df5.isnull().sum()

df5['N'] = df5['N'].ffill()



df5.plot()


training_set = df5.iloc[:542].values
test_set = df5.iloc[542:].values







