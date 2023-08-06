# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 13:45:18 2023

@author: Tanisha
"""

"""
ARIMA

"""
# Import Basic Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('preprocessed_p.csv', index_col = 'created_at', parse_dates = True)


df.plot()


train = df.iloc[:350000,0]
test = df.iloc[350000:,0]

# Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

Decomp_results = seasonal_decompose(df)

Decomp_results.plot()

Decomp_results.seasonal.plot()

# Finding the Parameters (p,d,q)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(train, lags=50)

plot_pacf(train, lags=50)


data4=df
import statsmodels.api as sm
q=1
p=1
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



print(test.mean()," and ",predictions.mean())

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test,predictions))
rmse 

'''
from pmdarima import auto_arima

auto_arima(df, trace = True)


# Developing ARIMA Model

from statsmodels.tsa.arima_model import ARIMA
q=1
p=1
d=0


A_Model = ARIMA(train, order = (p,d,q))

predictor = A_Model.fit()

predictor.summary()

Predicted_results = predictor.predict(start = len(train), end = len(train)+len(test) - 1, typ = 'levels')

plt.plot(test, color = 'red', label = 'Actual Temp')
plt.plot(Predicted_results, color = 'blue', label = 'Predicted Temp')
plt.xlabel ('Day')
plt.ylabel('Temp')
plt.legend()
plt.show()

test.mean()

Predicted_results.mean()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test,Predicted_results))

'''


