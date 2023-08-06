# Import Basic Nackages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read data
df4 = pd.read_csv('datac.csv', index_col = 'created_at', parse_dates = True)

df4

df4 = df4.iloc[:, :-5]

df4

column_names=["entry_id","pH","Relative_Humidity","Temperature","EC","N","N","K","other","pH","Relative_Humidity","Temperature","EC","N","N","K"]
df4.columns = column_names

df41=df4.iloc[:,:-8]
#extract N
df41=df41['N']
df41=pd.DataFrame(df41)
df41= pd.to_numeric(df41['N'], errors='coerce')
df41=pd.DataFrame(df41)
df41['N']=df41['N'].mask(df41['N']>=60000,200)
type(df41['N'])
df42= pd.to_numeric(df41['N'], errors='coerce')
#df41['2022-08-15T18:02:10+05:30':'2022-08-15T18:04:12+05:30']=20


df42.plot()

df43 = df42[df42.index.year == 2022]
df43.plot()

#resampling
df44=df42.resample('H').mean()
df44=pd.DataFrame(df44)
df44=df44['N'].interpolate(method='linear')
df44.plot()
df44.fillna(df44.mean())
df44.isna().sum()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df44);
plot_pacf(df44);

#dividing into 2 sets
df46=df44[df44.index.year == 2022]

# course model
training_set = df46
test_set = df44[df44.index.year==2023]
training_set=training_set['2022-08-16 00:00:00+05:30':]
test_set=test_set[:'2023-05-09 23:00:00+05:30']
training_set=pd.DataFrame(training_set)
test_set=pd.DataFrame(test_set)

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
#plt.xlim([2022,2023])
plt.ylabel('N values')
plt.legend()
plt.show()














































'''

dfl2=df4.iloc[:10000,:-8]
dfl2

df4

#df4 = df4.iloc[1: , :]
#df4

#df4 = df4.iloc[:, :-5]
#df4

#df4=df4['N']
#type(df4.index)

dfl2n=dfl2['N']
dfl2n
type(dfl2n.index)

dfl2n.index.asfreq = 'H'

dfl2n = pd.DataFrame(dfl2n)

dfl2n
dfl2n= pd.to_numeric(dfl2n['N'], errors='coerce')
dfl2n = dfl2n.replace(65535, np.nan)
dfl2n.plot()
dfl2n = dfl2n.ffill()






dfl2nn=dfl2n[1:10000,:]

dfl2nn.plot()

df4['N'] = df4['N'].ffill()

df4= pd.to_numeric(df4['N'], errors='coerce')

df4

df4=df4.resample(rule='H').mean()

#df4.dropna(inplace = True)

df4
df4 = pd.DataFrame(df4)
df4.isnull().sum()

df4['N'] = df4['N'].ffill()



df4.plot()


training_set = df4.iloc[:542].values
test_set = df4.iloc[542:].values

'''





