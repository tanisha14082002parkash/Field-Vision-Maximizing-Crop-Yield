# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 14:54:28 2023

@author: Tanisha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df1 = pd.read_csv('datac.csv', index_col = 'created_at', parse_dates = True)



dff = df1.iloc[:, :-5]



column_names=["entry_id","pH","Relative_Humidity","Temperature","EC","N","P","K","other","pH","Relative_Humidity","Temperature","EC","N","P","K"]
dff.columns = column_names

df1=dff.iloc[:,:-8]


dff
df10=dff['K']
type(df10.index)


df10.index.asfreq = 'S'

df10 = pd.DataFrame(df10)

df10
df10['K'] = df10['K'].ffill()
df10= pd.to_numeric(df10['K'], errors='coerce')

df10

df10=df10.resample(rule='H').mean()

#df10.dropna(inplace = True)

df10
df10 = pd.DataFrame(df10)
df10.isnull().sum()

df10['K'] = df10['K'].ffill()



df11=dff['N']
type(df11.index)


df11.index.asfreq = 'S'

df11 = pd.DataFrame(df11)

df11
df11 = pd.DataFrame(df11)
df11= pd.to_numeric(df11['N'], errors='coerce')
df11.plot()

df11['N'] = df11['N'].ffill()
df11= pd.to_numeric(df11['N'], errors='coerce')

df11

df11=df11.resample(rule='H').mean()

#df11.dropna(inplace = True)

df11
df11 = pd.DataFrame(df11)
df11.isnull().sum()

df11['N'] = df11['N'].ffill()


df12=dff['P']
type(df12.index)


df12.index.asfreq = 'S'

df12 = pd.DataFrame(df12)

df12

df12= pd.to_numeric(df12['P'], errors='coerce')

df12
df12 = pd.DataFrame(df12)
df12['P'] = df12['P'].ffill()
df12=df12.resample(rule='H').mean()

#df12.dropna(inplace = True)

df12
df12 = pd.DataFrame(df12)
df12.isnull().sum()
df12 = pd.DataFrame(df12)
df12['P'] = df12['P'].ffill()

df13=dff['pH']

df13 = pd.DataFrame(df13)
df13= pd.to_numeric(df13['pH'], errors='coerce')
df13 = pd.DataFrame(df13)



df13.loc[df13['pH']>14,'pH']=pd.np.nan
type(df13.index)


df13.index.asfreq = 'S'

df13 = pd.DataFrame(df13)

df13
df13['pH'] = df13['pH'].ffill()
df13= pd.to_numeric(df13['pH'], errors='coerce')

df13

df13=df13.resample(rule='H').mean()

#df13.dropna(inplace = True)

df13
df13 = pd.DataFrame(df13)
df13.isnull().sum()

df13['pH'] = df13['pH'].ffill()

dataset=df10.join(df11)
dataset=dataset.join(df13)
gfg_csv_data = dataset.to_csv('fourparameters.csv', index = True)

dataset.plot()
data2=dataset[['pH','P','K']]


