#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
##pip install factor_analyzer
from factor_analyzer import FactorAnalyzer
##pip install pingouin
import pingouin as pg


# In[4]:


df_train = pd.read_csv('F:/New folder/IIT Assignments/Arilines/train.csv')
df_train.head()


# In[6]:



df_test = pd.read_csv('F:/New folder/IIT Assignments/Arilines/test.csv')
df_test.head()


# In[7]:



df_train = df_train.select_dtypes(exclude=(object,bool))


# In[8]:


df_test = df_test.select_dtypes(exclude=(object,bool))


# In[9]:


df_train.values.shape


# In[10]:


df_train.dropna()


# In[11]:


df_train.values.shape


# In[12]:


df_train.astype('float64')


# In[13]:


df_train.replace([np.inf, -np.inf], np.nan).dropna(axis=1)


# In[14]:


df_test.shape


# In[15]:


df_test.dropna()


# In[16]:


df_test.shape


# In[17]:


x =df_train[df_train.columns[6:20]] 
fa = FactorAnalyzer()
fa.fit(x, 10)
ev, v = fa.get_eigenvalues()
ev
plt.plot(range(1,x.shape[1]+1),ev)


# In[18]:



fa = FactorAnalyzer(3, rotation='varimax')
fa.fit(x)
loads = fa.loadings_
print(loads)


# In[19]:


factor1 = df_train[['Food and drink', 'Seat comfort', 'Inflight entertainment', 'Cleanliness']]
factor2 = df_train[['On-board service', 'Baggage handling', 'Inflight service']]
factor3 = df_train[['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location']]


# In[20]:


factor1_alpha = pg.cronbach_alpha(factor1)
factor2_alpha = pg.cronbach_alpha(factor2)
factor3_alpha = pg.cronbach_alpha(factor3)
print(factor1_alpha, factor2_alpha, factor3_alpha)


# In[ ]:




