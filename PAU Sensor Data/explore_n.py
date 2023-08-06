# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:47:34 2023

@author: Tanisha
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('preprocessed_n.csv', index_col = 'created_at', parse_dates = True)

import statsmodels.api as sm
from pylab import rcParams

rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(df, model='additive')
fig = decomposition.plot()
plt.show()

from statsmodels.tsa.stattools import adfuller
result = adfuller(df)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
 
train=df[:'2023-04-01 00:00:00+05:30']
test=df['2023-04-01 00:00:00+05:30':]
 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(train, lags=50)

plot_pacf(train, lags=50)



# Group the data by month
grouped = df.groupby(pd.Grouper(freq='M'))



# Iterate over each month and plot the data
for month, data in grouped:
    # Create a new figure and axis for each month's plot
    fig, ax = plt.subplots()
    
    # Plot the data for the current month
    ax.plot(data.index, data['N'])
    
    ax.set_title('Data for {}'.format(month.strftime('%B %Y')))
    ax.set_xlabel('Date')
    ax.set_ylabel('Data')
    
    # Alternatively, use plt.show() to display the plot
    plt.show()



import statsmodels.api as sm
q=2
p=3
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