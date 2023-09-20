#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


# In[2]:


df=pd.read_csv("NIFTY50_all.csv",index_col=0,parse_dates=True)


# In[3]:


df


# In[4]:


nf=df[(df['Symbol']=="ONGC")]


# In[5]:


nf


# In[6]:


nf.columns


# In[7]:


nf1=nf[['High','Low']]


# In[8]:


nf1


# In[9]:


nf1['High'].plot()


# In[10]:


nf1


# In[11]:


nf1.info()


# In[12]:


nf1.describe()


# In[13]:


nf1['diff']=nf1['Low']-nf1['Low'].shift(1)


# In[14]:


nf1


# In[15]:


nf1['diff'].dropna().plot()


# In[16]:


nf1['diff2']=nf1['Low']-nf1['Low'].shift(2)


# In[17]:


nf1


# In[18]:


nf1['diff2'].dropna().plot()


# In[19]:


from statsmodels.tsa.stattools import adfuller


# In[20]:


adfuller(nf1['diff'].dropna())


# In[21]:


adfuller(nf1['diff2'].dropna())


# In[22]:


def adf_test(series):
    result=adfuller(series)
    print('ADF Statistics: {}'.format(result[0]))
    print('p- value: {}'.format(result[1]))
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis, reject the null hypothesis.indicating it is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")


# In[23]:


adf_test(nf1['diff2'].dropna())


# In[24]:


import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 


# In[25]:


fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig=sm.graphics.tsa.plot_acf(nf1['diff2'].dropna(),lags=40,ax=ax1)
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(nf1['diff2'].dropna(),lags=40,ax=ax2)


# In[26]:


from statsmodels.tsa.arima.model import ARIMA


# In[27]:


model=ARIMA(nf1['diff2'],order=(3,2,1))
model_fit=model.fit()


# In[28]:


model_fit.summary()


# In[39]:


len=nf1['diff2'].dropna()


# In[40]:


len


# In[41]:


train=nf1[:len-1000]
test=nf1[len-1000:]


# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


#import statsmodels.api as sm


# In[31]:


#model=sm.tsa.statespace.SARIMAX(nf1['diff2'],order=(3,2,1),seasonal_order=(3,2,1,12))
#results=model.fit()


# In[32]:


#nf1['Forecast']=model_fit.predict(start=len(train),end=nf1[nf1['diff2'].dropna()-1000:],dynamic=True)


# In[ ]:





# In[ ]:





# In[33]:


#len=len(nf1['diff2'].dropna())


# In[34]:


#len(nf1['diff2'])


# In[35]:


#train=nf1[:len-1000]


# In[36]:


#train.shape


# In[37]:


#test=nf1[len-1000:]


# In[38]:


#test.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




