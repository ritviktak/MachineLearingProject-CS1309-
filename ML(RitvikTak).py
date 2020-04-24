#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\RITVIK TAK\Downloads\ritvik.csv')


# In[34]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.describe()


# In[8]:


df.shape


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


df.keys()


# In[12]:


print(df['Y'])


# In[14]:


from sklearn.preprocessing import StandardScaler


# In[17]:


df.dtypes 


# In[46]:


scaler = StandardScaler()
df.iloc[:,1:] = scaler.fit_transform(df.iloc[:,1:])


# In[47]:


df


# In[48]:


from sklearn.decomposition import PCA


# In[49]:


pca = PCA(n_components=2)


# In[50]:


pca.fit(df.iloc[:,1:])


# In[51]:


x_pca = pca.transform(df.iloc[:,1:])


# In[52]:


df.iloc[:,1:].shape


# In[53]:


x_pca.shape


# In[72]:


from sklearn.preprocessing import LabelEncoder
# creating initial dataframe
Y = ('t','n')
#Y_df = pd.DataFrame(Y, columns=['Y'])
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
df['Y_cats'] = labelencoder.fit_transform(df['Y'])


# In[74]:


df[['Y_cats']]


# In[76]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=df['Y_cats'] ,cmap="plasma")
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')


# In[ ]:




