#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import shap

shap.initjs()


# In[2]:



#pip install shap


# In[3]:


X,y = shap.datasets.adult()
X_display,y_display = shap.datasets.adult(display=True)
X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=7)


# In[4]:


knn = sklearn.neighbors.KNeighborsClassifier()
knn.fit(X_train, y_train)


# In[ ]:





# In[5]:


f = lambda x: knn.predict_proba(x)[:,1]
med = X_train.median().values.reshape((1,X_train.shape[1]))
explainer = shap.KernelExplainer(f, med)
shap_values_single = explainer.shap_values(X.iloc[0,:], nsamples=1000)
shap.force_plot(explainer.expected_value, shap_values_single, X_display.iloc[0,:])


# In[6]:


shap_values = explainer.shap_values(X_valid.iloc[0:1000,:], nsamples=1000)
shap.force_plot(explainer.expected_value, shap_values, X_valid.iloc[0:1000,:])


# In[7]:


shap.summary_plot(shap_values, X_valid.iloc[0:1000,:])


# In[8]:


# normalize data
dtypes = list(zip(X.dtypes.index, map(str, X.dtypes)))
X_train_norm = X_train.copy()
X_valid_norm = X_valid.copy()
for k,dtype in dtypes:
    m = X_train[k].mean()
    s = X_train[k].std()
    X_train_norm[k] -= m
    X_train_norm[k] /= s
    
    X_valid_norm[k] -= m
    X_valid_norm[k] /= s


# In[9]:


knn_norm = sklearn.neighbors.KNeighborsClassifier()
knn_norm.fit(X_train_norm, y_train)


# In[10]:


f = lambda x: knn_norm.predict_proba(x)[:,1]
med = X_train_norm.median().values.reshape((1,X_train_norm.shape[1]))
explainer = shap.KernelExplainer(f, med)
shap_values_norm = explainer.shap_values(X_valid_norm.iloc[0:1000,:], nsamples=1000)
shap.force_plot(explainer.expected_value, shap_values_norm, X_valid.iloc[0:1000,:])


# In[11]:


shap.summary_plot(shap_values_norm, X_valid.iloc[0:1000,:])


# In[12]:


shap.dependence_plot("Education-Num", shap_values_norm, X_valid.iloc[0:1000,:])


# In[13]:


shap.dependence_plot("Education-Num", shap_values, X_valid.iloc[0:1000,:])


# In[ ]:




