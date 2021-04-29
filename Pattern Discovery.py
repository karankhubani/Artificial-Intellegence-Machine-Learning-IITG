#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
# !pip install plotnine
from plotnine import *
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
import time


# In[ ]:


df = pd.read_csv('D:/Data/EFA_AirLine/train.csv')
df.head()


# In[ ]:


df[df.satisfaction=='neutral or dissatisfied']=1
df[df.satisfaction=='satisfied']=0


# In[ ]:


df.columns


# In[ ]:


drop_cols = ["Unnamed: 0" , "id"]
num_cols = ["Age" , "Flight Distance" ,"Departure Delay in Minutes" , "Arrival Delay in Minutes"]
y_col = "satisfaction"
cat_cols = list(set(df.columns).difference(set(drop_cols+num_cols)))


# In[ ]:



ohe = OneHotEncoder(drop="first")
cat_df = pd.DataFrame(ohe.fit_transform(df[cat_cols]).todense() , columns=ohe.get_feature_names(cat_cols))


# In[ ]:



mms = MinMaxScaler()
num_df = pd.DataFrame(mms.fit_transform(df[num_cols]) , columns=num_cols)


# In[ ]:


##Now we join our categorical dataframe with our numerical dataframe.


# In[4]:


X = cat_df.join(num_df)
y = df[y_col]


# In[2]:


X = X.fillna(X.mean()) ### replacing missing values with mean


# In[3]:


X_train , X_test , y_train  ,y_test = train_test_split(X[:10000],y[:10000],test_size=.33,random_state=42)


# In[7]:


kmeans = KMeans(2)
kmeans.fit(X_train)


# In[8]:


pred = kmeans.predict(X)
pred


# In[9]:


y


# In[10]:



c = 0
for i,j in zip(pred,y) :
    if i == j :
        c += 1
print(c*100/len(y))


# In[11]:


logging.info('TSNE started')


# In[ ]:




