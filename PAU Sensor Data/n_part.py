# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:48:44 2023

@author: Tanisha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df11 = pd.read_csv('preprocessed_n.csv', index_col = 'created_at', parse_dates = True)
df1=df11.copy()
df1=df1['2022-12-01 00:00:00+05:30':'2023-01-31 23:00:00+05:30']

import statsmodels.api as sm
from pylab import rcParams

rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(df1, model='additive')
fig = decomposition.plot()
plt.show()

df1.plot()

from statsmodels.tsa.stattools import adfuller
result = adfuller(df1)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
 
df2=df1
df2['N_sma'] = df2.N.rolling(10, min_periods=1).mean()

colors = ['green', 'red']
# Line plot 
df2.plot(color=colors, linewidth=3, figsize=(12,6))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Simple Moving Average', fontsize=20)
plt.xlabel('date', fontsize=16)
plt.ylabel('N', fontsize=16)

df2['CMA'] = df2.N.expanding().mean()
# green -Avg Air Temp and Orange -CMA
colors = ['green', 'orange']
# line plot
df2[['N', 'CMA']].plot(color=colors, linewidth=3, figsize=(12,6))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels =['N', 'CMA'], fontsize=14)
plt.title('cumulative moving average', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('N', fontsize=16)


df2['EMA_0.1'] = df2.N.ewm(alpha=0.1, adjust=False).mean()
# Let's smoothing factor  - 0.3
df2['EMA_0.3'] = df2.N.ewm(alpha=0.3, adjust=False).mean()
# green - Avg Air Temp, red- smoothing factor - 0.1, yellow - smoothing factor  - 0.3
colors = ['green', 'red', 'yellow']
df2[['N', 'EMA_0.1', 'EMA_0.3']].plot(color=colors, linewidth=3, figsize=(12,6), alpha=0.8)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels=['N', 'EMA - alpha=0.1', 'EMA - alpha=0.3'], fontsize=14)
plt.title('Exponential Moving Average', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('N', fontsize=16)

df3=df3['2022-12-01 00:00:00+05:30':'2023-01-31 23:00:00+05:30']
train= df3[:'2023-01-02 00:00:00+05:30']
test=df3['2023-01-02 00:00:00+05:30':]


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df3=df11.copy()
plot_acf(df3)
plt.show()
plot_acf(df2['N_sma'])
plot_acf(df2['CMA'])
plot_acf(df2['EMA_0.1'])
plot_acf(df2['EMA_0.3'])





from matplotlib import pyplot
from pandas.plotting import lag_plot
lag_plot(df3)
pyplot.show()



#import libraries
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
# train autoregression
model = AutoReg(train, lags=20)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
# Predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


#calculate first order differencing value
first_order_diff=df3.N.diff()
#remove na value due to differencing
first_order_diff=first_order_diff.dropna()
#ADF test
result = adfuller(first_order_diff)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
 print('\t%s: %.3f' % (key, value))

first_order_diff.plot()
train= first_order_diff[:'2023-01-02 00:00:00+05:30']
test=first_order_diff['2023-01-02 00:00:00+05:30':]

#arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(train, lags=50)

plot_pacf(train, lags=50)



import statsmodels.api as sm
q=1
p=1
d=1
order = (p, d, q)  # Replace p, d, q with the appropriate values for your ARIMA model
# p: the number of autoregressive terms (lags)
# d: the degree of differencing required to make the time series stationary
# q: the number of moving average terms

# Create and fit the ARIMA model
model = sm.tsa.arima.ARIMA(train, order=order)
model_fit = model.fit()

# Print the summary of the model
print(model_fit.summary())

# Make predictions
start_index = len(train)  # Index from which predictions should start
end_index = len(train)+len(test) - 1  # Index at which predictions should end
predictions = model_fit.predict(start=start_index, end=end_index)

# Optionally, you can plot the actual data and the predictions
import matplotlib.pyplot as plt

plt.plot(test, label='Actual')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()


print(test.mean()," and ",predictions.mean())

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test,predictions))
rmse 

#implement lstm on this
train=pd.DataFrame(train)
test=pd.DataFrame(test)
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))

training_set_scaled = sc.fit_transform(train)
test_set_scaled = sc.fit_transform(test)

X_train = []
y_train = []
WS = 24

for i in range(WS, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-WS:i, 0:1])
    y_train.append(training_set_scaled[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)
    
# Developing LSTM Model

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

Model_P = Sequential()

Model_P.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1],1)))
Model_P.add(Dropout(0.2))

Model_P.add(LSTM(units = 60, return_sequences = True))
Model_P.add(Dropout(0.2))

Model_P.add(LSTM(units = 60, return_sequences = True))
Model_P.add(Dropout(0.2))

Model_P.add(LSTM(units = 60))
Model_P.add(Dropout(0.2))

Model_P.add(Dense(units = 1))

Model_P.compile(optimizer = 'adam', loss = 'mean_squared_error')

Model_P.fit(X_train, y_train, epochs = 30, batch_size = 24)

Model_P.save('LSTM - UniVariat')

from keras.models import load_model
Model_P = load_model('LSTM - UniVariat')


prediction_test = []

Batch_one = training_set_scaled[-WS:]
Batch_New = Batch_one.reshape((1,WS,1))

for i in range(720):
    
    First_Pred = Model_P.predict(Batch_New)[0]
    
    prediction_test.append(First_Pred)
    
    Batch_New = np.append(Batch_New[:,1:,:], [[First_Pred]], axis = 1)
    

prediction_test = np.array(prediction_test)

predictions = sc.inverse_transform(prediction_test)


plt.plot(test, color = 'red', label = 'Actual Values')
plt.plot(predictions, color = 'blue', label = 'predicted Values')
plt.title('LSTM - Univariate Forecast')
plt.xlabel('Time (h)')
plt.ylabel('P values')
plt.legend()
plt.show()



import math

from sklearn.metrics import mean_squared_error

RMSE = math.sqrt(mean_squared_error(test, predictions))

from sklearn.metrics import r2_score

Rsquare = r2_score(test_set, predictions)