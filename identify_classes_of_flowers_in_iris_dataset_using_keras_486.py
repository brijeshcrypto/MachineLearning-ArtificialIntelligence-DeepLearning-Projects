#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# In[3]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# In[4]:


import matplotlib
import matplotlib.pyplot as plt


# In[10]:


iris = load_iris()


# In[13]:


X = iris.data
y = iris.target


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[20]:


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


# In[22]:


model = keras.models.Sequential([
    keras.layers.Dense(300, input_shape=(4,), activation="relu"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(500, activation="relu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(3, activation="softmax")
])


# In[26]:


model.summary()


# In[30]:


model.compile(optimizer=keras.optimizers.Adam(lr=0.001), 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])


# In[33]:


history = model.fit(X_train, y_train, batch_size=5 , epochs=100)


# In[34]:


pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


# In[38]:


results = model.evaluate(X_test,y_test)


# In[39]:


print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))


# In[43]:


class_names = iris.target_names


# In[44]:


class_names


# In[45]:


X_new = X_test[:5]


# In[46]:


X_new


# In[47]:


y_pred = model.predict_classes(X_new)
print(np.array(class_names)[y_pred])


# In[48]:


y_new = y_test[:5]
print(np.array(class_names)[y_new])


# In[ ]:




