# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:36:19 2023

@author: Tanisha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df3=pd.read_csv('preprocessed_k.csv', index_col = 'created_at', parse_dates = True)
df3.plot()
#parts of df3
df=df3['2022-12-01 00:00:00+05:30':'2023-01-31 23:00:00+05:30']
df.plot()

from statsmodels.tsa.seasonal import seasonal_decompose
Decomp_results = seasonal_decompose(df)
Decomp_results.plot()
Decomp_results.seasonal.plot()

#calculate first order differencing value
first_order_diff=df.K.diff()
#remove na value due to differencing
first_order_diff=first_order_diff.dropna()
df=first_order_diff
df.plot()
Decomp_results = seasonal_decompose(df)
Decomp_results.plot()
Decomp_results.seasonal.plot()

train= df[:'2023-01-02 00:00:00+05:30']
test=df['2023-01-02 00:00:00+05:30':]
train.plot()
test.plot()

#arima with differencing
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

#arima without differencing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(train, lags=50)
plot_pacf(train, lags=50)


import statsmodels.api as sm
q=4
p=2
d=0
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

#sarima
order = (p, d, q)
P=1
D=1
Q=1
seasonal_order = (P, D, Q, 2)

sarima_model = sm.tsa.SARIMAX(train, order=order, seasonal_order=seasonal_order)
results = sarima_model.fit()

#lstm
train= df[:'2023-01-20 00:00:00+05:30']
test=df['2023-01-20 00:00:00+05:30':]
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))

training_set_scaled = sc.fit_transform(train)
test_set_scaled = sc.fit_transform(test)
plt.plot(training_set_scaled)
training_set_scaled=training_set_scaled[282:]

X_train = []
y_train = []
WS = 10

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

plt.plot(range(len(Model_P.history.history['loss'])),Model_P.history.history['loss'] )
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.show()


prediction_test = []

Batch_one = training_set_scaled[-WS:]
Batch_New = Batch_one.reshape((1,WS,1))

for i in range(288):
    
    First_Pred = Model_P.predict(Batch_New)[0]
    
    prediction_test.append(First_Pred)
    
    Batch_New = np.append(Batch_New[:,1:,:], [[First_Pred]], axis = 1)
    

prediction_test = np.array(prediction_test)

predictions = sc.inverse_transform(prediction_test)

test_arr = test['K'].values
plt.plot(test_arr, color = 'red', label = 'Actual Values')
plt.plot(predictions, color = 'blue', label = 'predicted Values')
plt.title('LSTM - Univariate Forecast')
plt.xlabel('Time (h)')
plt.ylabel('K values')
plt.legend()
plt.show()

import math

from sklearn.metrics import mean_squared_error

RMSE = math.sqrt(mean_squared_error(test, predictions))

from sklearn.metrics import r2_score

Rsquare = r2_score(test, predictions)
RMSE
Rsquare