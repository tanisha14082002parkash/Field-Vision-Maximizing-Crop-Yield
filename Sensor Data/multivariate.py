# -*- coding: utf-8 -*-
"""
Created on Wed May  3 17:02:23 2023

@author: Tanisha
"""

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

df1=df

df=df1
df.index.asfreq = 'S'

df = pd.DataFrame(df)

df
df['Temperature'] = df['Temperature'].ffill()
df= pd.to_numeric(df['Temperature'], errors='coerce')

df

df=df.resample(rule='H').mean()

#df.dropna(inplace = True)

df

df.isnull().sum()
df = pd.DataFrame(df)
df['Temperature'] = df['Temperature'].ffill()



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
df = df.reset_index(drop=True)

train = df.iloc[:542]
test = df.iloc[542:]

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 15:37:20 2023

@author: Tanisha
"""

"""
LSTM - Single Variate

"""
# Preprocessing

df1 = pd.read_csv('fourparameters.csv')

df1.dropna(inplace=True)
df1=df1['N']

dataset=df.join(df1)

df22=pd.read_csv('rh.csv')
df22.isnull().sum()
df22=df22['Relative_Humidity']
df22 = pd.DataFrame(df22)
dataset=dataset.join(df22)


training_set = dataset.iloc[:875,0:4].values
test_set = dataset.iloc[875:899,0:4].values


import seaborn as sn
sn.heatmap(dataset.corr())
dataset.plot()

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))

training_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.fit_transform(test_set)

test_set_scaled = test_set_scaled[:, 0:2]

X_train = []
y_train = []
WS = 24

for i in range(WS, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-WS:i, 0:3])
    y_train.append(training_set_scaled[i,2])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 3))

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout


Model = Sequential()

Model.add(LSTM(units = 70, return_sequences = True, input_shape = (X_train.shape[1], 3)))
Model.add(Dropout(0.2))

Model.add(LSTM(units = 70, return_sequences = True))
Model.add(Dropout(0.2))

Model.add(LSTM(units = 70, return_sequences = True))
Model.add(Dropout(0.2))

Model.add(LSTM(units = 70))
Model.add(Dropout(0.2))

Model.add(Dense(units = 1))

Model.compile(optimizer = 'adam', loss = 'mean_squared_error')

Model.fit(X_train,y_train, epochs = 80, batch_size = 32)

plt.plot(range(len(Model.history.history['loss'])), Model.history.history['loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.show()

Model.save('LSTM - Multivariate')

from keras.models import load_model
Model = load_model('LSTM - Multivariate')

prediction_test = []

Batch_one = training_set_scaled[-24:]
Batch_New = Batch_one.reshape((1,24,3))

for i in range(24):
    
    First_pred = Model.predict(Batch_New)[0]
    
    prediction_test.append(First_pred)
    
    New_var = test_set_scaled[i,:]
    
    New_var = New_var.reshape(1,2)
    
    New_test = np.insert(New_var, 2, [First_pred], axis =1)
    
    New_test = New_test.reshape(1,1,3)
    
    Batch_New = np.append(Batch_New[:,1:,:], New_test, axis=1)
    
prediction_test = np.array(prediction_test)

SI = MinMaxScaler(feature_range = (0,1))
y_Scale = training_set[:,2:3]
SI.fit_transform(y_Scale)

predictions = SI.inverse_transform(prediction_test)

real_values = test_set[:, 2]

plt.plot(real_values, color = 'red', label = 'Actual values')
plt.plot(predictions, color = 'blue', label = 'Predicted Values')
plt.title('Field Parameters')
#plt.xlabel('')
#plt.ylabel('Electrical Demand (MW)')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
RMSE = math.sqrt(mean_squared_error(real_values,predictions))


def mean_absolute_percentage_error (y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true)) * 100

MAPE = mean_absolute_percentage_error(real_values,predictions)
print(MAPE)