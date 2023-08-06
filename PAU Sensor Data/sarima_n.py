# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:01:25 2023

@author: Tanisha
"""

"""
SARIMAX

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

df = pd.read_csv('preprocessed_n.csv', index_col = 'created_at', parse_dates = True)
#df=df['2022-12-01 00:00:00+05:30':'2023-01-31 23:00:00+05:30']
df.plot()
decomposition = seasonal_decompose(
df['N'], model='additive', period=744)

decomposition.plot()
plt.show()

train=df[:'2023-04-01 00:00:00+05:30']
test=df['2023-04-01 00:00:00+05:30':]

model = SARIMAX(
train['N'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 744))

result = model.fit()
plt.plot(test)
plt.plot(result, color='red')
plt.show()








import seaborn as sn
sn.heatmap(df.corr())

train = df.iloc[:510, 0]
test = df.iloc[510:, 0]

exo = df.iloc[:,1:4]
exo_train = exo.iloc[:510]
exo_test = exo.iloc[510:]

from statsmodels.tsa.seasonal import seasonal_decompose

Decomp_results = seasonal_decompose(df['N'])

Decomp_results.plot()

from pmdarima import auto_arima

auto_arima(df['N'], exogenous = exo, m = 7, trace = True, D=1).summary()


from statsmodels.tsa.statespace.sarimax import SARIMAX

Model = SARIMAX(train, exog = exo_train, order = (2,0,2), seasonal_order = (0,1,1,7) )

Model = Model.fit()

prediction = Model.predict(len(train), len(train)+len(test)-1, exog = exo_test, typ = 'levels' )

plt.plot(test, color = 'red', label = 'Actual Temp')
plt.plot(prediction, color = 'blue', label = 'Predicted Temp')
plt.xlabel('Day')
plt.ylabel('Temp')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test,prediction))


