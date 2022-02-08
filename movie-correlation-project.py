#!/usr/bin/env python
# coding: utf-8

# # movie-industry
# 
# Use the "Run" button to execute the code.

# In[40]:


get_ipython().system('pip install jovian --upgrade --quiet')
#import libraries
import jovian
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8) #Adjusting the configurations of charts that we'll create

# reading the data

df = pd.read_csv('movies.csv')


# In[50]:


# Taking a look at the data

df.head(4)


# In[3]:


df.shape


# In[4]:


# Looking at the missing data

df.isnull().mean()


# In[42]:


#dropping the null data

df1 = df.dropna()


# In[6]:


# datatypes 

df1.dtypes


# In[56]:


#change datatype of columns
df1["budget"] = df1['budget'].astype(np.int64)
df1["gross"] = df1['gross'].astype(np.int64)


# In[57]:


df1.sample(3)


# In[9]:


pd.set_option('display.max_rows', None)


# In[10]:


df1 = df1.sort_values("gross",inplace = False, ascending = False)


# In[11]:


df1.sample(5)


# In[12]:


df.company.unique


# In[13]:


# Changing the gross value from scientific to normal

pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[51]:


df1.head(5)


# In[17]:


# plot the budget vs gross using seaborn

sns.regplot(x='budget', y='gross', data = df1, scatter_kws = {"color": "red"},line_kws = {"color":"green"})
plt.title('Budget Vs Gross earnings');


# In[44]:


# looking at correlation

df1.corr()


# In[ ]:


# High correlation between budget and gross


# In[45]:


correlation = df1.corr()

sns.heatmap(correlation, annot = True, cmap = 'Blues')
plt.title('Correlation Matrics')
plt.xlabel('Features')
plt.ylabel('Budget for film');


# In[53]:


# Giving numeric value to company for further analysis

dfrm_numeric = df1.copy()

for column in dfrm_numeric.columns:
    if(dfrm_numeric[column].dtypes == 'object'):
        dfrm_numeric[column]=dfrm_numeric[column].astype('category')
        dfrm_numeric[column]=dfrm_numeric[column].cat.codes

dfrm_numeric.head()


# In[49]:


correlation =dfrm_numeric.corr()

sns.heatmap(correlation, annot = True, cmap = 'Blues')
plt.title('Correlation Matrics')
plt.xlabel('Features')
plt.ylabel('Features');


# * There is good correlation between votes and budget also 
# * Among all the other features the Budget and gross income are highly correlated

# In[ ]:


jovian.commit(project = 'movie_industry')

