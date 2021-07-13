#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv("https://raw.githubusercontent.com/AP-State-Skill-Development-Corporation/Datasets/master/Regression/FuelConsumptionCo2.csv")


# In[2]:


data


# In[4]:


data.head()


# In[5]:


data.isnull().sum()


# In[6]:


data.shape


# In[7]:


data["MODELYEAR"].value_counts()


# In[8]:


x=data[["FUELCONSUMPTION_CITY","FUELCONSUMPTION_HWY","FUELCONSUMPTION_COMB"]]

y=data["CO2EMISSIONS"]


# In[26]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
x_train


# In[27]:


x_test


# In[28]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# In[29]:


y_pred=model.predict(x_test)
y_pred


# In[30]:


model.predict([[13.7,8.7,11.5]])


# In[31]:


from sklearn.metrics import r2_score,mean_squared_error
r2_score(y_test,y_pred)*100


# In[33]:


mean_squared_error(y_test,y_pred)**0.5


# In[34]:


model.intercept_


# In[35]:


model.coef_


# # POLYNOMIAL REGRESSION

# In[44]:


data=pd.read_csv("https://raw.githubusercontent.com/AP-State-Skill-Development-Corporation/Datasets/master/Regression/china_gdp.csv")


# In[45]:


data.shape


# In[46]:


data.head()


# In[47]:


data.shape


# In[48]:


import matplotlib.pyplot as plt
plt.scatter(data["Year"],data["Value"])
plt.show()


# In[58]:


x=data["Year"].values.reshape(-1,1)
y=data["Value"].values


# In[59]:


from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=3)
x_poly=poly.fit_transform(x)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_poly,y)


# In[60]:


y_pred=model.predict(x_poly)


# In[61]:


r2_score(y,y_pred)*100


# In[62]:


mean_squared_error(y,y_pred)


# In[65]:


plt.scatter(data["Year"],data["Value"],color="blue")
plt.plot(x,y_pred,color="red")
plt.show()


# In[72]:


acc=[]
for i in range(2,20):
    poly=PolynomialFeatures(degree=i)
    x_poly=poly.fit_transform(x)
    from sklearn.linear_model import LinearRegression
    model=LinearRegression()
    model.fit(x_poly,y)
    y_pred=model.predict(x_poly)
    acc.append(r2_score(y,y_pred)*100)


# In[73]:


acc


# In[76]:


import numpy as np
plt.plot(np.arange(2,20),acc)
plt.xlabel("Degree of polynomial")
plt.ylabel("Accuracy")
plt.show()


# In[ ]:




