# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:05:30 2023

@author: aviral
"""

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

df51=df5.iloc[:,:-8]
df51=df51['P']
df51=pd.DataFrame(df51)
df52= pd.to_numeric(df51['P'], errors='coerce')
df51['2022-08-15T18:02:10+05:30':'2022-08-15T18:04:12+05:30']=20
df51['P']=df51['P'].mask(df51['P']>=200,20)
type(df51['P'])
df52.plot()

df53 = df52[df52.index.year == 2022]
df53.plot()

df53 = df53.resample('H').mean()
df53.plot()
df53=pd.DataFrame(df53)
df53= df53['P'].interpolate(method='linear')
df53.plot()

df54=df52.resample('H').mean()
df54=pd.DataFrame(df54)
df54=df54['P'].interpolate(method='linear')
df54.plot()
df54.fillna(df54.mean())

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df54);
plot_pacf(df54);

df56=df54[df54.index.year == 2022]

# course model
training_set = df56
test_set = df54[df54.index.year==2023]
training_set=training_set['2022-08-16 00:00:00+05:30':]
test_set=test_set[:'2023-05-09 23:00:00+05:30']

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))

training_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.fit_transform(test_set)

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

Model_P.fit(X_train, y_train, epochs = 30, batch_size = 32)

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

for i in range(3096):
    
    First_Pred = Model_P.predict(Batch_New)[0]
    
    prediction_test.append(First_Pred)
    
    Batch_New = np.append(Batch_New[:,1:,:], [[First_Pred]], axis = 1)
    

prediction_test = np.array(prediction_test)

predictions = sc.inverse_transform(prediction_test)


plt.plot(test_set, color = 'red', label = 'Actual Values')
plt.plot(predictions, color = 'blue', label = 'predicted Values')
plt.title('LSTM - Univariate Forecast')
plt.xlabel('Time (h)')
plt.ylabel('P values')
plt.legend()
plt.show()

import math

from sklearn.metrics import mean_squared_error

RMSE = math.sqrt(mean_squared_error(test_set, predictions))

from sklearn.metrics import r2_score

Rsquare = r2_score(test_set, predictions)

'''
#df5 = df5.iloc[1: , :]
#df5

#df5 = df5.iloc[:, :-5]
#df5

#df5=df5['N']
#type(df5.index)

dfl1n=dfl1['P']
dfl1n
type(dfl1n.index)

dfl1n.index.asfreq = 'H'

dfl1n = pd.DataFrame(dfl1n)

dfl1n
dfl1n= pd.to_numeric(dfl1n['P'], errors='coerce')
dfl1n = dfl1n.replace(65535, np.nan)

dfl1n = dfl1n.ffill()
dfl1n.plot()

def sampling(sequence, n_steps):
    x,y=list(),list()
    for i in range(len(sequence)):
        sam=i+n_steps
        if sam>len(sequence)-1:
            break
        xx,yy=sequence[i:sam],sequence[sam]
        x.append(xx)
        y.append(yy)
    return np.array(x),np.array(y)


n_steps=3
df54=pd.DataFrame(df54)
df54.isna().sum()
df54.fillna(df54.mean())
x,y=sampling(df54['P'].tolist(),n_steps)
np.isnan(x).sum()
np.nan_to_num(x, nan=20.0)
np.isnan(y).sum()
for i in range(len(x)):
    print(x[i],y[i])

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
x=x.reshape((x.shape[0],x.shape[1],1))
model.summary()
model.fit(x,y, epochs=50)

df56.isna().sum()
x_test,y_test=sampling(df56['P'].tolist(),n_steps)
x_test = x_test.reshape((x_test.shape[0], n_steps, 1))
loss = model.evaluate(x_test,y_test)
y_pred = model.predict(x_test)
y_pred
plt.plot(y_pred, label='Predicted')
# Plotting the actual values
plt.plot(y_test, label='Actual')
# Set labels and title
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Predicted vs Actual')
# Add legend
plt.legend()
# Display the plot
plt.show()
'''