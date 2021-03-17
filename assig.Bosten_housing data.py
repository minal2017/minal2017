#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("housingdatas.csv")


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.isnull().sum()/df.shape[0]*100


# In[7]:


df.isnull().sum(axis=1)/df.shape[1]*100


# In[8]:


A=list(df[((df.isnull().sum(axis=1)/df.shape[1])*100)>5].index)


# In[9]:


print(A)


# In[10]:


for i in A:#remove >5 % row parmanantly
    df.drop(i,inplace=True)


# In[11]:


df.shape


# In[12]:


#check again null value
df.isnull().sum()


# In[13]:


#visualize null value 
sns.heatmap(df.isnull())


# In[14]:


df.info()


# In[15]:


#check correlation
X=df.drop("MEDV",axis=1)
Y=df["MEDV"]


# In[16]:


X.columns


# In[17]:


for col in X:
    sns.scatterplot(data=df,x=col,y=Y)
    plt.show()


# In[18]:


plt.figure (figsize=(9,9))
plt.show()


# In[19]:


X=df.drop("MEDV",axis=1)
Y=df["MEDV"]


# In[20]:


#action the model inuilt function
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)


# In[21]:


from sklearn.linear_model import LinearRegression


# In[22]:


#train the model with the help of Linear Regression
lr=LinearRegression()


# In[23]:


#train the model
lr.fit(X_train,Y_train)


# In[24]:


lr.coef_


# In[25]:


lr.intercept_


# In[26]:


Y_pred=lr.predict(X_test)


# In[27]:


print(Y_pred)


# In[28]:


from sklearn.metrics import mean_squared_error


# In[29]:


mse=mean_squared_error(Y_test,Y_pred)
rmse=np.sqrt(mse)
print("root means squared error,",rmse)


# In[30]:


from sklearn.metrics import r2_score
r2_score(Y_test,Y_pred)


# In[31]:


X=df[["LSTAT","RM","PTRATIO","TAX","INDUS"]]
Y=df["MEDV"]


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)


# In[34]:


lr=LinearRegression()


# In[35]:


lr.fit(X_train,Y_train)


# In[36]:


lr.coef_


# In[37]:


lr.intercept_


# In[38]:


Y_pred=lr.predict(X_test)


# In[39]:


Y_pred


# In[40]:


mse=mean_squared_error(Y_test,Y_pred)
print("mean squared error",mse)
rmes=np.sqrt(mse)
print("root mean squared error",rmse)


# In[41]:


r2_score(Y_test,Y_pred)


# In[42]:


residuals=Y_test-Y_pred


# In[43]:


sns.scatterplot(Y_pred,residuals)
plt.show()


# In[44]:


#normal distrubuted curve
sns.distplot(residuals)
plt.show()


# In[45]:


#increse degree of x
#we perform polynomials regression
from sklearn.preprocessing import PolynomialFeatures


# In[59]:


#pass degree of x
pf=PolynomialFeatures(2)


# In[60]:


X=df[["LSTAT","RM","PTRATIO","TAX","INDUS"]]#input always 2D
Y=df["MEDV"]#target means output


# In[61]:


X_poly=pf.fit_transform(X)


# In[62]:


X_train,X_test,Y_train,Y_test=train_test_split(X_poly,Y,test_size=0.3,random_state=1)


# In[63]:


lr=LinearRegression()


# In[64]:


lr.fit(X_train,Y_train)


# In[65]:


Y_pred=lr.predict(X_test)


# In[66]:


Y_pred


# In[67]:


mse=mean_squared_error(Y_pred,Y_test)


# In[68]:


mse


# In[69]:


r2_score(Y_pred,Y_test)


# In[70]:


residuals=Y_test-Y_pred
sns.scatterplot(Y_pred,residuals)
plt.show()


# In[71]:


#normal distrubuted curve
sns.distplot(residuals)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




