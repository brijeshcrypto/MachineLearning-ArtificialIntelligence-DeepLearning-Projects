#!/usr/bin/env python
# coding: utf-8

# In[1]:


# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"


# Common imports
import numpy as np
import os


# In[2]:


def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

def plot_color_image(image):
    plt.imshow(image, interpolation="nearest")
    plt.axis("off")


# In[3]:


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3 as myModel
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions


# In[4]:


model = myModel(weights="imagenet")


# In[11]:


img_path = 'cloudxlab_jupyter_notebooks/ml/elephant.jpg'


# In[13]:


import matplotlib.pyplot as plt
img = image.load_img(img_path, target_size=(299, 299))

print ("img.shape", type(img))
plt.imshow(np.asarray(img)) #To enable showing PIL image


# In[14]:



#Converts a PIL image to np array
x = image.img_to_array(img)
print ("x.shape", x.shape)


# In[15]:


#We have to feed an array of images 
x= np.array([x])
#x = np.expand_dims(x, axis=0)
print ("x expand dims shape", x.shape)


# In[16]:


#Preprocess the color channels for the specific model
x = preprocess_input(x)
print ("x expand preprocess", x.shape)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])


# In[ ]:




