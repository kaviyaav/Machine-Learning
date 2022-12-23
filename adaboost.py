#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import os


# In[3]:


train = pd.read_csv("train.csv")
train.head()


# In[5]:


test = pd.read_csv("test.csv")
test.head()


# In[6]:


all = pd.concat([train, test], sort = False)
all.info()


# In[7]:


#Fill Missing numbers with median
all['Age'] = all['Age'].fillna(value=all['Age'].median())
all['Fare'] = all['Fare'].fillna(value=all['Fare'].median())


# In[8]:


all.info()


# In[9]:


sns.catplot(x = 'Embarked', kind = 'count', data = all) #or all['Embarked'].value_counts()


# In[10]:


all['Embarked'] = all['Embarked'].fillna('S')
all.info()


# In[11]:


#Age
all.loc[ all['Age'] <= 16, 'Age'] = 0
all.loc[(all['Age'] > 16) & (all['Age'] <= 32), 'Age'] = 1
all.loc[(all['Age'] > 32) & (all['Age'] <= 48), 'Age'] = 2
all.loc[(all['Age'] > 48) & (all['Age'] <= 64), 'Age'] = 3
all.loc[ all['Age'] > 64, 'Age'] = 4 


# In[12]:


#Title
import re
def get_title(name):
    title_search = re.search(' ([A-Za-z]+\.)', name)
    
    if title_search:
        return title_search.group(1)
    return ""


# In[13]:


all['Title'] = all['Name'].apply(get_title)
all['Title'].value_counts()


# In[14]:


all['Title'] = all['Title'].replace(['Capt.', 'Dr.', 'Major.', 'Rev.'], 'Officer.')
all['Title'] = all['Title'].replace(['Lady.', 'Countess.', 'Don.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Royal.')
all['Title'] = all['Title'].replace(['Mlle.', 'Ms.'], 'Miss.')
all['Title'] = all['Title'].replace(['Mme.'], 'Mrs.')
all['Title'].value_counts()


# In[15]:


#Cabin
all['Cabin'] = all['Cabin'].fillna('Missing')
all['Cabin'] = all['Cabin'].str[0]
all['Cabin'].value_counts()


# In[16]:


#Family Size & Alone 
all['Family_Size'] = all['SibSp'] + all['Parch'] + 1
all['IsAlone'] = 0
all.loc[all['Family_Size']==1, 'IsAlone'] = 1
all.head()


# In[17]:


#Drop unwanted variables
all_1 = all.drop(['Name', 'Ticket'], axis = 1)
all_1.head()


# In[18]:


all_dummies = pd.get_dummies(all_1, drop_first = True)
all_dummies.head()


# In[19]:


all_train = all_dummies[all_dummies['Survived'].notna()]
all_train.info()


# In[20]:


all_test = all_dummies[all_dummies['Survived'].isna()]
all_test.info()


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_train.drop(['PassengerId','Survived'],axis=1), 
                                                    all_train['Survived'], test_size=0.30, 
                                                    random_state=101, stratify = all_train['Survived'])


# In[23]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# In[24]:


ada = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=100, random_state=0)
ada.fit(X_train,y_train)


# In[25]:


predictions = ada.predict(X_test)


# In[26]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[27]:


print (f'Train Accuracy - : {ada.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {ada.score(X_test,y_test):.3f}')


# In[28]:


TestForPred = all_test.drop(['PassengerId', 'Survived'], axis = 1)


# In[29]:


t_pred = ada.predict(TestForPred).astype(int)


# In[30]:


PassengerId = all_test['PassengerId']


# In[31]:


adaSub = pd.DataFrame({'PassengerId': PassengerId, 'Survived':t_pred })
adaSub.head()


# In[ ]:




