#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
data = pd.read_csv('F:/New folder/IIT Assignments/Bag of Words/Clustering_gmm.csv')
import matplotlib.pyplot as plt 
plt.figure(figsize=(7,7))
plt.scatter(data["Weight"],data["Height"])
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Data Distribution')
plt.show()


# In[11]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(data)


# In[12]:


pred = kmeans.predict(data)
frame = pd.DataFrame(data)
frame['cluster'] = pred
frame.columns = ['Weight', 'Height', 'cluster']


# In[13]:


color=['blue','green','cyan', 'black']
for k in range(0,4):
    data = frame[frame["cluster"]==k]
    plt.scatter(data["Weight"],data["Height"],c=color[k])
plt.show()


# In[14]:


import pandas as pd
data = pd.read_csv('F:/New folder/IIT Assignments/Bag of Words/Clustering_gmm.csv')


# In[15]:


from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4)
gmm.fit(data)


# In[16]:


labels = gmm.predict(data)
frame = pd.DataFrame(data)
frame['cluster'] = labels
frame.columns = ['Weight', 'Height', 'cluster']


# In[10]:


color=['blue','green','cyan', 'black']
for k in range(0,4):
    data = frame[frame["cluster"]==k]
    plt.scatter(data["Weight"],data["Height"],c=color[k])


# In[ ]:




