#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[24]:


dataset = pd.read_csv('50_Startups.csv')


# In[36]:


x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(y)


# In[26]:


# encoding of Independent Variable


# In[27]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))


# In[29]:


# Splitting dataset


# In[30]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 1)


# In[31]:


# Training the multiple linear regression on training set


# In[32]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[ ]:


# predicting the test set results


# In[40]:


y_pred = regressor.predict(x_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_test.reshape(len(y_test),1), y_pred.reshape(len(y_pred),1)),1))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




