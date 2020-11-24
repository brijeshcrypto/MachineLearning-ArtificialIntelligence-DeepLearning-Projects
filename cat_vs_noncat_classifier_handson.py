#!/usr/bin/env python
# coding: utf-8

# In[117]:


import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# In[118]:


# Loading training Data

train_dataset = h5py.File('/cxldata/datasets/project/cat-non-cat/train_catvnoncat.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # train set features
train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # train set labels


# In[143]:


train_set_y_orig


# In[119]:


train_set_x_orig.shape


# In[120]:


# Load test data
test_dataset = h5py.File('/cxldata/datasets/project/cat-non-cat/test_catvnoncat.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # test set features
test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # test set labels


# In[121]:


test_set_x_orig/255


# In[122]:


classes = np.array(test_dataset["list_classes"][:])


# In[123]:


classes.shape


# In[124]:


print(test_set_x_orig.shape)
print(test_set_y_orig.shape)


# In[148]:


# Standardize data to have feature values between 0 and 1.
X_train = train_set_x_orig/255.
X_test = test_set_x_orig/255.

print ("train_x's shape: " + str(X_train.shape))
print ("test_x's shape: " + str(X_test.shape))
print ("test_y shape: " + str(test_set_y.shape))
print ("train_y shape: " + str(train_set_y.shape))


# In[149]:


train_x_flatten.shape


# In[150]:


# Reshape the train and test set labels
train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))


# In[151]:


print(test_set_y)
print(test_set_y.shape)


# In[152]:


# Example of a picture

index = 50
plt.imshow(train_set_x_orig[index]) # You should see a cat image
y = train_set_y[:, index]
print(y.shape)
y_class = classes[np.squeeze(train_set_y[:, index])].decode("utf-8")
print(y)
print(y_class)


# In[153]:


train_set_x_orig.shape


# In[156]:


import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Conv2D,InputLayer, Dropout, BatchNormalization, Flatten, Dense, MaxPooling2D
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

tf.keras.backend.clear_session()

CNNModel = tf.keras.Sequential([
    L.InputLayer(input_shape=(64,64,3)),
    L.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    L.BatchNormalization(),
    L.MaxPooling2D((2, 2)),
    L.Conv2D(64, (3, 3), activation='relu'),
    L.MaxPooling2D((2, 2)),
    L.Flatten(),
    L.Dense(64, activation='relu'),
    L.Dropout(rate=0.5),
    L.Dense(1, activation='sigmoid')
])

sgd = tf.keras.optimizers.SGD(learning_rate=0.007)

CNNModel.compile(optimizer='sgd',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])


# In[157]:


CNNModel.summary()


# In[ ]:





# In[158]:


history = CNNModel.fit(X_train, train_set_y_orig, epochs=50, validation_split=0.2, batch_size=64)


# In[159]:


import pandas as pd
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
#plt.gca().set_ylim(0.03, 0.05) # setting limits for y-axis
plt.show()


# In[160]:


loss, acc = CNNModel.evaluate(X_test,test_set_y_orig,verbose=0)
print('Test loss: {}'.format(loss))
print('Test Accuracy: {}'.format(acc))


# In[161]:


test_set_y_pred = CNNModel.predict(X_test)


# In[162]:


print(np.round(test_set_y_pred))


# In[163]:


def plot(X,y):
    plt.imshow(X.reshape(64,64,3))
    plt.show()


# In[166]:


n=3
plot(test_set_x_orig[n],test_set_y_orig_pred[n])


# In[ ]:




