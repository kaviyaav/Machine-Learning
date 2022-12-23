#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


dataset = pd.read_csv('Cellphone.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : ,  4 ].values


# In[4]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[ : , 3])
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()


# In[5]:


X = X[: , 1:]


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[8]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# In[9]:


y_pred = regressor.predict(X_test)


# In[11]:


Accuracy=r2_score(Y_test,y_pred)*100
print(" Accuracy of the model is %.2f" %Accuracy)


# In[13]:


plt.scatter(Y_test,y_pred);
plt.xlabel('Actual');
plt.ylabel('Predicted');


# In[15]:


sns.regplot(x=Y_test,y=y_pred,ci=None,color ='red');


# In[16]:


pred_df=pd.DataFrame({'Actual Value':Y_test,'Predicted Value':y_pred,'Difference':Y_test-y_pred})


# In[17]:


pred_df


# In[ ]:




