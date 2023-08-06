# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:08:36 2023

@author: Tanisha
"""


# Import Basic Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df4 = pd.read_excel('Sensor Data-October 02 2022.xlsx', index_col = 'created_at', parse_dates = True)

df4


column_names=["entry_id","pH","Relative_Humidity","Temperature","EC","N","P","K","other"]
df4.columns = column_names

df4

df4 = df4.iloc[1: , :]

df4

df4 = df4.iloc[:, :-1]

df4

df4=df4['EC']
type(df4.index)


df4.index.asfreq = 'S'

df4 = pd.DataFrame(df4)

df4
df4['EC'] = df4['EC'].ffill()
df4= pd.to_numeric(df4['EC'], errors='coerce')

df4

df4=df4.resample(rule='T').mean()

#df4.dropna(inplace = True)

df4
df4 = pd.DataFrame(df4)
df4.isnull().sum()

#df4['EC'] = df4['EC'].ffill()





# find the indices of the missing values
missing_idx = df4['EC'].index[df4['EC'].isnull()]

# loop through the missing indices and fill each gap separately
for i, idx in enumerate(missing_idx):
    if i % 2 == 0:  # fill with ffill
        prev_idx = missing_idx[i-1] if i > 0 else None
        next_idx = missing_idx[i+1] if i < len(missing_idx)-1 else None
        df4.loc[idx:next_idx, 'A'] = df4.loc[idx:next_idx, 'A'].ffill(limit=next_idx-idx+1)
    else:  # fill with bfill
        prev_idx = missing_idx[i-1]
        next_idx = missing_idx[i+1] if i < len(missing_idx)-1 else None
        df4.loc[prev_idx+1:idx, 'A'] = df4.loc[prev_idx+1:idx, 'A'].bfill(limit=idx-prev_idx)
        



df4.plot()


training_set = df4.iloc[:542].values
test_set = df4.iloc[542:].values



