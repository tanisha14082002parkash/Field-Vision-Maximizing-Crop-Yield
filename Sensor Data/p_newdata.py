# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:05:30 2023

@author: aviral
"""

# Import Basic Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df5 = pd.read_csv('datac.csv', index_col = 'created_at', parse_dates = True)

df5

df5 = df5.iloc[:, :-5]

df5

column_names=["entry_id","pH","Relative_Humidity","Temperature","EC","N","P","K","other","pH","Relative_Humidity","Temperature","EC","N","P","K"]
df5.columns = column_names

dfl1=df5.iloc[:10000,:-8]
dfl1

df5

#df5 = df5.iloc[1: , :]
#df5

#df5 = df5.iloc[:, :-5]
#df5

#df5=df5['N']
#type(df5.index)

dfl1n=dfl1['P']
dfl1n
type(dfl1n.index)

dfl1n.index.asfreq = 'H'

dfl1n = pd.DataFrame(dfl1n)

dfl1n
dfl1n= pd.to_numeric(dfl1n['P'], errors='coerce')
dfl1n = dfl1n.replace(65535, np.nan)

dfl1n = dfl1n.ffill()
dfl1n.plot()
