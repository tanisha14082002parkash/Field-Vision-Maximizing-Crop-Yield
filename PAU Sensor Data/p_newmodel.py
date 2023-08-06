# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:44:02 2023

@author: Tanisha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read dataset
data = pd.read_csv('preprocessed_p.csv', index_col = 'created_at', parse_dates = True)

df4=data['2022-12-15 00:00:00+05:30':'2023-01-15 23:00:00+05:30']
df4.plot()

train= df4[:'2023-01-10 00:00:00+05:30']
test=df4['2023-01-10 00:00:00+05:30':]

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))

training_set_scaled = sc.fit_transform(train)
test_set_scaled = sc.fit_transform(test)
plt.plot(training_set_scaled)
#training_set_scaled=training_set_scaled[282:]

X_train = []
y_train = []
WS = 10

for i in range(WS, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-WS:i, 0:1])
    y_train.append(training_set_scaled[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout




model = Sequential()
model.add(LSTM(200, input_shape=(X_train.shape[1],1))) #,return_sequences = True,activation='sigmoid')
model.add(Dropout(0.2))
#model.add(LSTM(60))
#.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train, epochs=30, batch_size=24, shuffle=False)

#------------------------------------------

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

Model_P.fit(X_train, y_train, epochs = 30, batch_size = 24, shuffle=False)


prediction_test = []

Batch_one = training_set_scaled[-WS:]
Batch_New = Batch_one.reshape((1,WS,1))

for i in range(144):
    
    First_Pred = model.predict(Batch_New)[0]
    
    prediction_test.append(First_Pred)
    
    Batch_New = np.append(Batch_New[:,1:,:], [[First_Pred]], axis = 1)
    

prediction_test = np.array(prediction_test)

predictions = sc.inverse_transform(prediction_test)


test_arr = test['P'].values
plt.plot(test_arr, color = 'red', label = 'Actual Values')
plt.plot(predictions, color = 'blue', label = 'predicted Values')
plt.title('LSTM - Univariate Forecast with 200 neurons')
plt.xlabel('Time (min)')
plt.ylabel('P values')
plt.legend()
plt.show()
