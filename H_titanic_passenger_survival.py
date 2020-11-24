#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# In[3]:


TITANIC_PATH = '/cxldata/datasets/project/titanic'


# In[10]:


def load_titanic_data(filename,titanic_path = TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)


# In[11]:


train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")


# In[12]:


train_data.head(5)


# In[20]:


train_data.info()


# In[17]:


train_data.shape


# In[21]:


train_data.describe()


# In[24]:


train_data["Sex"].value_counts()[1]


# In[18]:


test_data.shape


# In[19]:


test_data.head(3)


# In[27]:


from sklearn.base import BaseEstimator,TransformerMixin


# In[30]:


class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


# In[39]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
    ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
    ("imputer", SimpleImputer(strategy="median")),
])


# In[40]:


num_pipeline.fit_transform(train_data,test_data)


# In[41]:


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


# In[45]:


from sklearn.preprocessing import OneHotEncoder


# In[46]:


cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])


# In[47]:


cat_pipeline.fit_transform(train_data,test_data)


# In[50]:


from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# In[51]:


X_train = preprocess_pipeline.fit_transform(train_data)


# In[52]:


y_train = train_data["Survived"]


# In[56]:


from sklearn.svm import SVC


# In[57]:


svm_clf  = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_train, y_train)


# In[60]:


X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)


# In[63]:


y_pred.shape


# In[64]:


from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()


# In[67]:


svm_scores


# In[70]:


from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()


# In[ ]:





# In[33]:


train_data.head(2)


# In[ ]:





# In[ ]:





# In[ ]:


train

