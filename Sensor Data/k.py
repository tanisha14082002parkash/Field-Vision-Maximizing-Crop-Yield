
# Import Basic Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df7 = pd.read_excel('Sensor Data-October 02 2022.xlsx', index_col = 'created_at', parse_dates = True)

df7


column_names=["entry_id","pH","Relative_Humidity","Temperature","EC","N","P","K","other"]
df7.columns = column_names

df7

df7 = df7.iloc[1: , :]

df7

df7 = df7.iloc[:, :-1]

df7

df7=df7['K']
type(df7.index)


df7.index.asfreq = 'S'

df7 = pd.DataFrame(df7)

df7
df7['K'] = df7['K'].ffill()
df7= pd.to_numeric(df7['K'], errors='coerce')

df7

df7=df7.resample(rule='H').mean()

#df7.dropna(inplace = True)

df7
df7 = pd.DataFrame(df7)
df7.isnull().sum()

df7['K'] = df7['K'].ffill()



df7.plot()


#training_set = df7.iloc[:542].values
#test_set = df7.iloc[542:].values







