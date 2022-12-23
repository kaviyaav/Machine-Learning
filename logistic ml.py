#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[23]:


dataset = pd.read_csv('healthcare.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[26]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[27]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# In[28]:


y_pred = classifier.predict(X_test)


# In[29]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[42]:


pd.crosstab(dataset.heart_disease,dataset.gender).plot(kind='bar')
plt.title('Heart disease')
plt.xlabel('Gender')
plt.ylabel('Frequency of heart disease')
plt.savefig('purchase_fre_job')


# In[35]:


dataset.age.hist()
plt.title('Histogram of Age')
plt.xlabel('age')
plt.ylabel('heart_disease')
plt.savefig('hist_age')


# In[36]:


import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


# In[37]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[38]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[39]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[40]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[41]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[ ]:




