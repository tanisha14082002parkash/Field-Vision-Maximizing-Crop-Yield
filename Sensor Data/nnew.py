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

dfl1n=dfl1['N']
dfl1n
type(dfl1n.index)

dfl1n.index.asfreq = 'H'

dfl1n = pd.DataFrame(dfl1n)

dfl1n
dfl1n= pd.to_numeric(dfl1n['N'], errors='coerce')
dfl1n = dfl1n.replace(65535, np.nan)
dfl1n.plot()
dfl1n = dfl1n.ffill()






dfl1nn=dfl1n[1:10000,:]

dfl1n10000.plot()

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







