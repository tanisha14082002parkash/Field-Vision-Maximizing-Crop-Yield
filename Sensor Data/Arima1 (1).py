# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import Basic Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_excel('Sensor Data-October 02 2022.xlsx', index_col = 'created_at', parse_dates = True)

df


column_names=["entry_id","pH","Relative_Humidity","Temperature","EC","N","P","K","other"]
df.columns = column_names

df

df = df.iloc[1: , :]

df

df = df.iloc[:, :-1]

df

type(df.index)

df.index.asfreq = 'S'

df = pd.DataFrame(df['Temperature'])

df

df.dropna(inplace = True)

df

df.isnull().sum()


df= pd.to_numeric(df['Temperature'], errors='coerce')

df

df.plot()


train = df.iloc[:58236]
test = df.iloc[58236:]

train

df = df.dropna()
df.isnull().sum()

df.index.asfreq = 'S'
df=df.resample(rule='H').mean()
df = df.dropna()
df.isnull().sum()

train = df.iloc[:542]
test = df.iloc[542:]


# Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

df.sort_index(inplace=True)

Decomp_results = seasonal_decompose(df,period=200)
#41592



Decomp_results.plot()
# Finding the Parameters (p,d,q)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(train, lags=50)

plot_pacf(train, lags=50)


from pmdarima import auto_arima

auto_arima(df, trace = True)

# Developing ARIMA Model

from statsmodels.tsa.arima.model import ARIMA

A_Model = ARIMA(train, order = (0,1,2))

predictor = A_Model.fit()

predictor.summary()

Predicted_results = predictor.predict(start = len(train), end = len(train)+len(test) - 1,typ='levels')

plt.plot(test, color = 'red', label = 'Actual Temp')
plt.plot(Predicted_results, color = 'blue', label = 'Predicted Temp')
plt.xlabel ('hr')
plt.ylabel('Temp')
plt.legend()
plt.show()

test.mean()

Predicted_results.mean()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test,Predicted_results))




