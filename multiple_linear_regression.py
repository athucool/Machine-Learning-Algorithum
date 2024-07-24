#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd


# In[71]:


import numpy as np


# In[72]:


data=pd.read_csv("C:\\Users\\athar\\Documents\\data science\\mlr\\multiple_linear_regression.txt")


# In[73]:


data.head()


# In[75]:


data.loc[:,['Feature 1','Feature 2']]


# In[76]:


data['Target'] 


# In[77]:


data.shape


# In[78]:


output_col='Target'


# In[79]:


x=data.iloc[:,data.columns!=output_col]


# In[80]:


x


# In[81]:


from  sklearn.model_selection import train_test_split
 
 


# In[82]:


y=data.loc[:,output_col]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=43)


# In[83]:


y_test.shape 


# In[84]:


x_test.shape


# In[85]:


#Linear regression

from sklearn.linear_model import LinearRegression


# In[91]:


lr=LinearRegression()


# In[92]:


lr.fit(x_train,y_train)


# In[88]:


lr.coef_


# In[98]:


lr.intercept_


# In[99]:


x_test


# In[120]:


predict_value=lr.predict(x_test)


# In[121]:


predict_value[:]


# In[122]:


from sklearn.metrics import mean_squared_error


# In[123]:


cost=mean_squared_error(y_test,predict_value)


# In[124]:


cost


# In[125]:


import matplotlib.pyplot as plt
plt.plot(x_test,y_test,'+',color="green")
plt.plot(x_test,predict_value,'*',color='red')
xlabel='input'
ylabel='output'
plt.show()


# In[126]:


residual=y_test-predict_value


# In[127]:


plt.scatter(residual,y_test)
plt.title("residual vs y_test")
 
    
plt.show()


# In[128]:


import seaborn as sns


# In[129]:


sns.distplot(residual)


# In[130]:


import statsmodels.api as sm
x_withconstant=sm.add_constant(x_train)


# In[131]:


x_withconstant


# In[138]:


#ordinary least squar
model=sm.OLS(y_train,x_withconstant)


# In[139]:


model


# In[140]:


result=model.fit()


# In[141]:


result.summary()


# In[142]:


result.params


# In[143]:


lr.intercept_

