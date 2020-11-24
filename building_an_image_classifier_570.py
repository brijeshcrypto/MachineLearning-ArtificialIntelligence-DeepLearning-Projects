#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing the necessary modules.
import numpy as np
import tensorflow as tf
print("Tensorflow version", tf.__version__)
from tensorflow import keras
print("Keras version", keras.__version__)
import matplotlib.pyplot as plt


# In[3]:


#Set the random seed for tf and np so as to reuse same set of random variables and reproduce the results.

np.random.seed(42)
tf.random.set_seed(42)


# In[10]:


#Loading the Dataset

fashion_mnist = keras.datasets.fashion_mnist


# In[11]:


(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


# In[12]:


X_train_full.shape


# In[19]:


X_valid = X_train_full[:5000]/255.
y_valid = y_train_full[:5000]


# In[20]:


X_train = X_train_full[5000:]/255.
y_train = y_train_full[5000:]


# In[21]:


print("Train data shape:",X_train.shape)
print("Validation data shape:",X_valid.shape)
print("Test data shape:",X_test.shape)


# In[22]:


X_test = X_test / 255.


# In[31]:


class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# In[32]:


#Visualizing the Data

print("Class label is:", y_train[0])
print("Class name is:", class_names[y_train[0]])
plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()


# In[33]:


n_rows = 4
n_cols = 10
plt.figure(figsize=(15, 6))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index],cmap="binary")
        plt.axis('off')
        plt.title(class_names[y_train[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()


# In[37]:


#Building the Model

keras.backend.clear_session()

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.layers


# In[38]:


model.summary()


# In[39]:


#arcitecture of model and saved it into my_fashion_mnist_model.png file

keras.utils.plot_model(model, "my_fashion_mnist_model.png", show_shapes=True)


# In[40]:


sgd = keras.optimizers.SGD(learning_rate=0.01)


# In[43]:


model.compile(loss="sparse_categorical_crossentropy",
      optimizer=sgd,
      metrics=["accuracy"])


# In[47]:


#Fitting the Model

history = model.fit(X_train, y_train, epochs=30,
            validation_data=(X_valid, y_valid))


# In[48]:


history.params


# In[49]:


#print the name of the first hidden layer:

hidden1 = model.layers[1]
print(hidden1.name)


# In[50]:


weights, biases = hidden1.get_weights() # getting the weights and biases
print(weights.shape, weights)
print(biases)


# In[51]:


import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # setting limits for y-axis
plt.show()


# In[55]:


#Evaluating the Model Performance

model.evaluate(X_test, y_test)


# In[58]:


#let's predict and visualize the first 3 samples from the test data.

y_pred = model.predict_classes(X_test[:3])
print(y_pred)
print(class_names[index] for index in y_pred)


# In[59]:


plt.figure(figsize=(7, 3))

for index, image in enumerate(X_test[:3]):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary")
    plt.axis('off')
    plt.title(class_names[y_pred[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()


# In[ ]:




