
# Import Basic Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df6 = pd.read_excel('Sensor Data-October 02 2022.xlsx', index_col = 'created_at', parse_dates = True)

df6


column_names=["entry_id","pH","Relative_Humidity","Temperature","EC","N","P","K","other"]
df6.columns = column_names

df6

df6 = df6.iloc[1: , :]

df6

df6 = df6.iloc[:, :-1]

df6

df6=df6['P']
type(df6.index)


df6.index.asfreq = 'S'

df6 = pd.DataFrame(df6)

df6
df6['P'] = df6['P'].ffill()
df6= pd.to_numeric(df6['P'], errors='coerce')

df6

df6=df6.resample(rule='H').mean()

#df6.dropna(inplace = True)

df6
df6 = pd.DataFrame(df6)
df6.isnull().sum()

df6['P'] = df6['P'].ffill()



df6.plot()


#training_set = df6.iloc[:542].values
#test_set = df6.iloc[542:].values







