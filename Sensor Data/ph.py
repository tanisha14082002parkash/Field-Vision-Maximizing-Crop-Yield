
# Import Basic Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df3 = pd.read_excel('Sensor Data-October 02 2022.xlsx', index_col = 'created_at', parse_dates = True)

df3


column_names=["entry_id","pH","Relative_Humidity","Temperature","EC","N","P","K","other"]
df3.columns = column_names

df3

df3 = df3.iloc[1: , :]

df3

df3 = df3.iloc[:, :-1]

df3

df3=df3['pH']
df3.plot()
df3 = pd.DataFrame(df3)
df3= pd.to_numeric(df3['pH'], errors='coerce')
df3 = pd.DataFrame(df3)
df55=df3


df3.loc[df3['pH']>14,'pH']=pd.np.nan
type(df3.index)


df3.index.asfreq = 'S'

df3 = pd.DataFrame(df3)

df3
df3['pH'] = df3['pH'].ffill()
df3= pd.to_numeric(df3['pH'], errors='coerce')

df3

df3=df3.resample(rule='H').mean()

#df3.dropna(inplace = True)

df3
df3 = pd.DataFrame(df3)
df3.isnull().sum()

df3['pH'] = df3['pH'].ffill()



df3.plot()


training_set = df3.iloc[:542].values
test_set = df3.iloc[542:].values







