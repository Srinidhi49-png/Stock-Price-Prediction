import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM


# Get the data of the stock AAPL
data = yf.download ('AAPL','2013-01-01','2023-01-01')
plt.plot(data.Close)
ma100 = data.Close.rolling(100).mean()
ma100
plt.figure(figsize = (12,6))
plt.plot(data.Close)
plt.plot(ma100, 'r')
ma200 = data.Close.rolling(200).mean()
ma200
plt.figure(figsize = (12,6))
plt.plot(data.Close)
plt.plot(ma200, 'r')
plt.figure(figsize = (12,6))
plt.plot(data.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')


#spliting Data into Training and Testing
data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])
print(data_training.shape)
print(data_testing.shape)
from sklearn.preprocessing import MinMaxScaler


#scaling the data
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)
data_training_array


#training dataset
x_train = []
y_train = []
for i in range(100, data_training_array.shape[0]):
 x_train.append(data_training_array[i-100: i])
 y_train.append(data_training_array[i, 0])
 x_train,y_train = np.array(x_train),np.array(y_train)


#LSTM model
model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True,
 input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 60, activation = 'relu', return_sequences = True,))
model.add(Dropout(0.3))
model.add(LSTM(units = 80, activation = 'relu', return_sequences = True,))
model.add(Dropout(0.4))
model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1))


#model compilation
model.compile(optimizer = 'adam',loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 50)
model.save('my_model.keras')
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)
input_data


#testing dataset
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
 x_test.append(input_data[i-100: i])
 y_test.append(input_data[i, 0])
x_test,y_test = np.array(x_test), np.array(y_test)


#making predictions
y_predicted = model.predict(x_test)
scale_factor = 1/0.00712466
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor
plt.figure(figsize=(12,6))
plt.plot(y_test, 'b',label = 'Original Price')
plt.plot(y_predicted,'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()