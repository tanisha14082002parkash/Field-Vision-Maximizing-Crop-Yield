# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:17:09 2023

@author: Tanisha
"""

# Import Basic Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read dataset
data = pd.read_csv('datac.csv', index_col = 'created_at', parse_dates = True)

df = data.iloc[:, :-5]

df

column_names=["entry_id","pH","Relative_Humidity","Temperature","EC","N","P","K","other","pH","Relative_Humidity","Temperature","EC","N","P","K"]
df.columns = column_names

df=df.iloc[:,:-8]

df=df['P']
df=pd.DataFrame(df)
df= pd.to_numeric(df['P'], errors='coerce')
df=pd.DataFrame(df)
df['P']=df['P'].mask(df['P']>=200,20)

#type(df['P'])
df= pd.to_numeric(df['P'], errors='coerce')
#df['2022-08-5T8:02:0+05:30':'2022-08-5T8:0:2+05:30']=20


#df2.plot()

#resampling
df4=df.resample('T').mean()
df4=pd.DataFrame(df4)
df4=df4['P'].interpolate(method='linear')
#df4 = df4.fillna(method='ffill')
df4.plot()
df4= pd.to_numeric(df4['P'], errors='coerce')
df4.fillna(df4.mean())
df4.isna().sum()
df4=df4['2022-08-16 00:00:00+05:30':]
df4=df4[:'2023-05-09 23:00:00+05:30']

df4.to_csv('interpolate_p.csv', index=True)

df4=df4['2022-12-15 00:00:00+05:30':'2023-01-15 23:00:00+05:30']
df4.plot()

train= df4[:'2023-01-10 00:00:00+05:30']
test=df4['2023-01-10 00:00:00+05:30':]
train.plot()
test.plot()

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
    
# Developing LSTM Model
#model1

## for Deep-learing:
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
model.add(LSTM(100, input_shape=(X_train.shape[1],1), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(60, return_sequences = True))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train, epochs=30, batch_size=24, shuffle=False)

'''
prediction_test = []
Batch_one = training_set_scaled[-WS:]
Batch_New = Batch_one.reshape((1, WS, 1))

for i in range(8581):
    First_Pred = model.predict(Batch_New)[0, -1, 0]
    prediction_test.append(First_Pred)

    # Reshape First_Pred to have the same dimensions as Batch_New
    First_Pred = First_Pred.reshape((1, 1, 1))

    # Concatenate First_Pred with Batch_New along axis 1
    Batch_New = np.concatenate((Batch_New, First_Pred), axis=1)
'''

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
plt.title('LSTM - Univariate Forecast')
plt.xlabel('Time (min)')
plt.ylabel('P values')
plt.legend()
plt.show()

#-----------------------------------------------------------------------
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

Model_P.fit(X_train, y_train, epochs = 30, batch_size = 24, shuffle=False)

Model_P.save('LSTM - UniVariat')

from keras.models import load_model
Model_P = load_model('LSTM - UniVariat')



prediction_test = []

Batch_one = training_set_scaled[-WS:]
Batch_New = Batch_one.reshape((1,WS,1))

for i in range(8581):
    
    First_Pred = Model_P.predict(Batch_New)[0]
    
    prediction_test.append(First_Pred)
    
    Batch_New = np.append(Batch_New[:,1:,:], [[First_Pred]], axis = 1)
    

prediction_test = np.array(prediction_test)

predictions = sc.inverse_transform(prediction_test)

test_arr = test['P'].values
plt.plot(test_arr, color = 'red', label = 'Actual Values')
plt.plot(predictions, color = 'blue', label = 'predicted Values')
plt.title('LSTM - Univariate Forecast')
plt.xlabel('Time (min)')
plt.ylabel('P values')
plt.legend()
plt.show()

import math

from sklearn.metrics import mean_squared_error

RMSE = math.sqrt(mean_squared_error(test, predictions))

from sklearn.metrics import r2_score

Rsquare = r2_score(test, predictions)
RMSE
Rsquare

#-------------------------------------------------------------------------
















