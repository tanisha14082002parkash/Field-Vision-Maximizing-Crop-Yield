# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:05:30 2023

@author: aviral
"""

# Import Basic Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df3 = pd.read_csv('datac.csv', index_col = 'created_at', parse_dates = True)

df3

df3 = df3.iloc[:, :-5]

df3

column_names=["entry_id","pH","Relative_Humidity","Temperature","EC","N","P","K","other","pH","Relative_Humidity","Temperature","EC","N","P","K"]
df3.columns = column_names

dfl1=df3.iloc[:10000,:-8]
dfl1

df3
df31=df3.iloc[:,:-8]
#extract N
df31=df31['N']
df31=pd.DataFrame(df31)
df31= pd.to_numeric(df31['N'], errors='coerce')
df31=pd.DataFrame(df31)
df31['N']=df31['N'].mask(df31['N']>=60000,200)
type(df31['N'])
df32= pd.to_numeric(df31['N'], errors='coerce')
#df31['2022-08-15T18:02:10+05:30':'2022-08-15T18:04:12+05:30']=20

#df3 = df3.iloc[1: , :]
#df3

#df3 = df3.iloc[:, :-5]
#df3

#df3=df3['N']
#type(df3.index)


dfl1n=dfl1['K']
dfl1n
type(dfl1n.index)

dfl1n.index.asfreq = 'H'

dfl1n = pd.DataFrame(dfl1n)

dfl1n
dfl1n= pd.to_numeric(dfl1n['K'], errors='coerce')
dfl1n = dfl1n.replace(65535, np.nan)

dfl1n = dfl1n.ffill()
dfl1n.plot()
