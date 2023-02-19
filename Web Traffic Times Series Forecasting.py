#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

data = pd.read_csv(r"C:\Users\Gaurav\Downloads\Thecleverprogrammer.csv")
print(data.head())


# In[2]:


data["Date"] = pd.to_datetime(data["Date"], 
                              format="%d/%m/%Y")
print(data.info())


# In[3]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.plot(data["Date"], data["Views"])
plt.title("Daily Traffic of Thecleverprogrammer.com")
plt.show()


# In[4]:


plot_pacf(data["Views"], lags = 100)


# In[5]:


pd.plotting.autocorrelation_plot(data["Views"])


# In[12]:


p, d, q = 5, 1, 2
model=sm.tsa.statespace.SARIMAX(data['Views'],
                                order=(p, d, q),
                                seasonal_order=(p, d, q, 12))
model=model.fit()
print(model.summary())


# In[10]:


predictions = model.predict(len(data), len(data)+50)
print(predictions)


# In[11]:


data["Views"].plot(legend=True, label="Training Data", 
                   figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")


# In[ ]:



Gaurav6321/Gaurav6321-Code-clause_FEB_Web-Traffic-Times-Series-Forecasting

