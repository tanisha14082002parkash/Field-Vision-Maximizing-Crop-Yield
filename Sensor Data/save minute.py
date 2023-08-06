# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:33:15 2023

@author: Tanisha
"""

# Import Basic Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df100 = pd.read_excel('Sensor Data-October 02 2022.xlsx', index_col = 'created_at', parse_dates = True)

df100


column_names=["entry_id","pH","Relative_Humidity","Temperature","EC","N","P","K","other"]
df100.columns = column_names

df100

df100 = df100.iloc[1: , :]

df100

df100 = df100.iloc[:, :-1]

df=df100['N']

type(df.index)


df.index.asfreq = 'S'

df = pd.DataFrame(df)

df
df['N'] = df['N'].ffill()
df= pd.to_numeric(df['N'], errors='coerce')

df

df=df.resample(rule='T').mean()

#df.dropna(inplace = True)

df

df.isnull().sum()
df = pd.DataFrame(df)
df['N'] = df['N'].ffill()



df.plot()

ndata = df.to_csv('n-minute.csv', index = True)

