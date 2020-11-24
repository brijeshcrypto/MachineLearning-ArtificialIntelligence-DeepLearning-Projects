#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('/cxldata/datasets/project/ny_stock_prediction/prices-split-adjusted.csv', header = 0)


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.symbol.unique()


# In[6]:


df_yahoo = df[df['symbol']=='YHOO']


# In[7]:


df_yahoo.head(10)


# In[8]:


df_yahoo.shape


# In[9]:


df_yahoo.describe()


# In[10]:


df_yahoo.isnull().sum()


# In[11]:


df_yahoo['date'] = pd.to_datetime(df_yahoo['date'])


# In[12]:


print(df_yahoo.date.dtype)


# In[13]:


df_yahoo = df_yahoo.set_index("date")


# In[14]:


df_yahoo.head()


# In[15]:


print(df_yahoo.shape)


# In[16]:


yahoo_data = df_yahoo.asfreq('b')


# In[17]:


print(yahoo_data.shape)


# In[18]:


yahoo_data.isnull().sum()


# In[19]:


null_data = yahoo_data[yahoo_data.isnull().any(axis=1)]


# In[20]:


null_data.head()


# In[21]:


null_dates = null_data.index.tolist()


# In[22]:


import calendar
import datetime

holidays = []

for date in null_dates:
    week, day, month, year = date.weekday(), date.day, date.month, date.year
    week_day = calendar.day_name[week]

    if month==1:
        if day==1:
            # New year day
            holidays.append(date)
        elif day==2 and week_day=='Monday':
            # Observed New Year Day
            holidays.append(date)
        elif day>=15 and day<=21 and week_day=='Monday':
            # Martin Luther King, Jr. Day
            holidays.append(date)

    elif month==2:
        # Washington's Birthday
        if day>=15 and day<=21 and week_day=='Monday':
            holidays.append(date)

    elif month==5:
        # Memorial day
        if day>=25 and day<=31 and week_day=='Monday':
            holidays.append(date)

    elif month==7:
        # Independence day
        if day==4:
            holidays.append(date)
        # Observed Independence  Day
        elif day==5 and week_day=='Monday':
            holidays.append(date)
        elif day==3 and week_day=='Friday':
            holidays.append(date)

    elif month == 9:
        # Labour day
        if day>=1 and day<=7 and week_day=='Monday':
            holidays.append(date)

    elif month==11:
        # Thanksgiving Day
        if week_day=='Thursday' and day>=22 and day<=28:
            holidays.append(date)

    elif month==12:
        # Christmas Day
        if day==25:
            holidays.append(date)
        # Observed Christmas Day
        elif day==24 and week_day=='Friday':
            holidays.append(date)
        elif day==26 and week_day=='Monday':
            holidays.append(date)

good_fridays = [ datetime.date(2010,4,2), datetime.date(2011,4,22), datetime.date(2012,4,6), datetime.date(2013,3,29), datetime.date(2014,4,18), datetime.date(2015,4,3), datetime.date(2016,3,25) ]
holidays = holidays + [pd.to_datetime(date) for date in good_fridays]

non_holidays = [x for x in null_dates if x not in holidays]
print(non_holidays)


# In[23]:


print(yahoo_data.shape)


# In[24]:


modified_df = yahoo_data.drop(holidays)


# In[25]:


modified_df.shape


# In[26]:


print("Before filling missing values:\n", modified_df.isnull().sum())


# In[27]:


modified_df = modified_df.bfill(axis = 'rows')


# In[28]:


print("\nAfter filling missing values:\n",modified_df.isna().sum())


# In[29]:


def plotter(code):
    global closing_stock
    plt.subplot(211)
    company_close = modified_df[modified_df['symbol']==code]
    company_close = company_close.close.values.astype('float32')
    company_close = company_close.reshape(-1, 1)
    closing_stock = company_close
    plt.xlabel('Time')
    plt.ylabel(code + " close stock prices")
    plt.title('prices Vs Time')
    plt.grid(True)
    plt.plot(company_close , 'b')
    plt.show()

plotter("YHOO")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


n_train = int(len(closing_stock) * 0.80)
n_remaining = len(closing_stock) - n_train

n_val = int(n_remaining*0.50)
n_test = n_remaining - n_val 
print("Train samples:",n_train, 
      "Validation Samples:",n_val,
      "Test Samples:", n_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[48]:


2+3


# In[31]:


train_data = closing_stock[0:n_train]
print(train_data.shape)


# In[49]:


val_data = closing_stock[n_train:n_train+n_val]
print(val_data.shape)


# In[51]:


test_data = closing_stock[n_train+n_val:]
print(test_data.shape)


# In[52]:


from sklearn.preprocessing import MinMaxScaler


# In[53]:


scaler = MinMaxScaler(feature_range = (0,1))


# In[54]:


train = scaler.fit_transform(train_data)


# In[55]:


val = scaler.transform(val_data)


# In[56]:


test = scaler.transform(test_data)


# In[57]:


float(test.max())


# In[70]:


def create_dataset(data , n_features):
    dataX, dataY = [], []
    for i in range(len(data)-n_features-1):
        a = data[i:(i+n_features), 0]
        dataX.append(a)
        dataY.append(data[i + n_features, 0])
    return np.array(dataX), np.array(dataY)


# In[71]:


n_features = 2


# In[72]:


trainX, trainY = create_dataset(train, n_features)
valX, valY = create_dataset(val, n_features)
testX, testY = create_dataset(test, n_features)


# In[73]:


print(trainX.shape , trainY.shape , valX.shape , valY.shape, testX.shape , testY.shape)


# In[74]:


trainX = trainX.reshape(trainX.shape[0] , 1 ,trainX.shape[1])
valX = valX.reshape(valX.shape[0] , 1 ,valX.shape[1])
testX = testX.reshape(testX.shape[0] , 1 ,testX.shape[1])


# In[75]:


print(trainX.shape , trainY.shape , valX.shape , valY.shape, testX.shape , testY.shape)


# In[83]:


import tensorflow as tf
tf.random.set_seed(42)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


# In[84]:


model = keras.Sequential()


# In[85]:


# First GRU layer
model.add(layers.GRU(units=100, return_sequences=True, input_shape=(1,n_features), activation='tanh'))
model.add(layers.Dropout(0.2))

# Second GRU layer
model.add(layers.GRU(units=150, return_sequences=True, input_shape=(1,n_features), activation='tanh'))
model.add(layers.Dropout(0.2))

# Third GRU layer
model.add(layers.GRU(units=100, activation='tanh'))
model.add(layers.Dropout(0.2))

# The output layer
model.add(layers.Dense(units=1, kernel_initializer='he_uniform', activation='linear'))


# In[86]:


model.compile(loss='mean_squared_error', 
              optimizer=Adam(lr = 0.0005) , 
              metrics = ['mean_squared_error'])


# In[87]:


print(model.summary())


# In[88]:


history = model.fit(trainX,trainY,epochs=100,batch_size=128, 
                    verbose=1, validation_data = (valX,valY))


# In[94]:


import math

def model_score(model, X_train, y_train, X_val, y_val , X_test, y_test):
    print('Train Score:')
    train_score = model.evaluate(X_train, y_train, verbose=0)
    print("MSE: {:.5f} , RMSE: {:.2f}".format(train_score[0], math.sqrt(train_score[0])))

    print('Validation Score:')
    val_score = model.evaluate(X_val, y_val, verbose=0)
    print("MSE: {:.5f} , RMSE: {:.2f}".format (val_score[0], math.sqrt(val_score[0])))

    print('Test Score:')
    test_score = model.evaluate(X_test, y_test, verbose=0)
    print("MSE: {:.5f} , RMSE: {:.2f}".format (test_score[0], math.sqrt(test_score[0])))


model_score(model, trainX, trainY ,valX, valY , testX, testY)


# In[97]:


print(history.history.keys())


# In[98]:


plt.plot(history.history['loss'])  # plotting train loss
plt.plot(history.history['val_loss'])  # plotting validation loss

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[99]:


pred = model.predict(testX)


# In[100]:


pred = scaler.inverse_transform(pred)
print(pred[:10])


# In[101]:


testY_actual = testY.reshape(testY.shape[0] , 1)
testY_actual = scaler.inverse_transform(testY_actual)
print(testY_actual[:10])


# In[102]:


plt.plot(testY_actual , 'b')
plt.plot(pred , 'r')

plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.title('Check the performance of the model with time')
plt.legend(['Actual', 'Predicted'], loc='upper left')

plt.grid(True)
plt.show()


# In[ ]:




