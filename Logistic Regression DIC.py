#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import io
import requests
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
from pandas.api.types import CategoricalDtype

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score

get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading Data

# In[2]:


columns = ["age", "workClass", "fnlwgt", "education", "education-num",
           "marital-status", "occupation", "relationship", "race", "sex", 
           "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

train_data = pd.read_csv('adult.data', names = columns, sep=' *, *', na_values='?')
test_data = pd.read_csv('adult.test', names = columns, sep=' *, *', skiprows =1, na_values='?')


# In[3]:


train_data.head()


# In[4]:


test_data.head()


# # Exploratory Data Analysis
# #Cleaning the data

# In[5]:


train_data.info()


# In[6]:


test_data.info()


# # Handing Numerical Attributes

# In[7]:


num_attributes = train_data.select_dtypes(include=['int'])
print(num_attributes.columns)


# In[8]:


num_attributes.hist(figsize=(10,10))


# # Handling Categorical Attributes

# In[9]:


cat_attributes = train_data.select_dtypes(include=['object'])
print(cat_attributes.columns)


# In[10]:


sns.set(rc={'figure.figsize':(8,6)})
sns.countplot(y='workClass', hue='income', data = cat_attributes)


# # ColumnSelector Pipeline

# In[11]:


class ColumnsSelector(BaseEstimator, TransformerMixin):
  
  def __init__(self, type):
    self.type = type
  
  def fit(self, X, y=None):
    return self
  
  def transform(self,X):
    return X.select_dtypes(include=[self.type])


# # Numerical Data Pipeline

# In[12]:


num_pipeline = Pipeline(steps=[
    ("num_attr_selector", ColumnsSelector(type='int')),
    ("scaler", StandardScaler())
])


# In[13]:


class CategoricalImputer(BaseEstimator, TransformerMixin):
  
  def __init__(self, columns = None, strategy='most_frequent'):
    self.columns = columns
    self.strategy = strategy
    
    
  def fit(self,X, y=None):
    if self.columns is None:
      self.columns = X.columns
    
    if self.strategy is 'most_frequent':
      self.fill = {column: X[column].value_counts().index[0] for column in self.columns}
    else:
      self.fill ={column: '0' for column in self.columns}
      
    return self
      
  def transform(self,X):
    X_copy = X.copy()
    for column in self.columns:
      X_copy[column] = X_copy[column].fillna(self.fill[column])
    return X_copy


# In[14]:


class CategoricalEncoder(BaseEstimator, TransformerMixin):
  
  def __init__(self, dropFirst=True):
    self.categories=dict()
    self.dropFirst=dropFirst
    
  def fit(self, X, y=None):
    join_df = pd.concat([train_data, test_data])
    join_df = join_df.select_dtypes(include=['object'])
    for column in join_df.columns:
      self.categories[column] = join_df[column].value_counts().index.tolist()
    return self
    
  def transform(self, X):
    X_copy = X.copy()
    X_copy = X_copy.select_dtypes(include=['object'])
    for column in X_copy.columns:
      X_copy[column] = X_copy[column].astype({column: CategoricalDtype(self.categories[column])})
    return pd.get_dummies(X_copy, drop_first=self.dropFirst)
 


# In[15]:


cat_pipeline = Pipeline(steps=[
    ("cat_attr_selector", ColumnsSelector(type='object')),
    ("cat_imputer", CategoricalImputer(columns=['workClass','occupation', 'native-country'])),
    ("encoder", CategoricalEncoder(dropFirst=True))
])


# In[16]:


full_pipeline = FeatureUnion([("num_pipe", num_pipeline), ("cat_pipeline", cat_pipeline)])


# # Building the Model

# In[17]:


train_data.drop(['fnlwgt', 'education'], axis=1, inplace=True)
test_data.drop(['fnlwgt', 'education'], axis=1, inplace=True)


# # Preparing the data for training

# In[18]:


# copy the data before preprocessing
train_copy = train_data.copy()

# convert the income column to 0 or 1 and then drop the column for the feature vectors
train_copy["income"] = train_copy["income"].apply(lambda x:0 if x=='<=50K' else 1)

# creating the feature vector 
X_train = train_copy.drop('income', axis =1)

# target values
Y_train = train_copy['income']

print(X_train.columns)


# # Training the model

# In[19]:


# set parameter type_df as train for categorical encoder 
# we can set parameter using the name of the transformer while defining the pipeline
# syntax:  name_of_the_transformer__ = 

# pass the data through the full_pipeline
X_train_processed = full_pipeline.fit_transform(X_train)
print(X_train_processed.shape)


# In[20]:


model = LogisticRegression(random_state=0)
model.fit(X_train_processed, Y_train)


# In[21]:


model.coef_


# # Testing the model

# In[22]:


# take a copy of the test data set
test_copy = test_data.copy()

# convert the income column to 0 or 1
test_copy["income"] = test_copy["income"].apply(lambda x:0 if x=='<=50K.' else 1)

# separating the feature vecotrs and the target values
X_test = test_copy.drop('income', axis =1)
Y_test = test_copy['income']

X_test.columns


# In[23]:


# preprocess the test data using the full pipeline
# here we set the type_df param to 'test'
X_test_processed = full_pipeline.fit_transform(X_test)
X_test_processed.shape


# In[24]:


predicted_classes = model.predict(X_test_processed)
print(predicted_classes)


# # Model Evaluation

# In[25]:


accuracy_score(predicted_classes, Y_test.values)


# In[26]:


sns.set(rc={'figure.figsize':(8,6)})
cfm = confusion_matrix(predicted_classes, Y_test.values)
sns.heatmap(cfm, annot=True)
print(cfm)
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')


# # Cross Validation

# In[27]:


cross_val_model = LogisticRegression(random_state=0)
scores = cross_val_score(cross_val_model, X_train_processed, Y_train, cv=5)
print(scores)
print(np.mean(scores))


# # Fine Tuning the Model

# In[28]:


penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
random_state=[0]

# creating a dictionary of hyperparameters
hyperparameters = dict(C=C, penalty=penalty, random_state=random_state)


# In[29]:


clf = GridSearchCV(estimator = model, param_grid = hyperparameters, cv=5)
best_model = clf.fit(X_train_processed, Y_train)
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])


# In[30]:


best_predicted_values = best_model.predict(X_test_processed)
print(best_predicted_values)


# In[31]:


accuracy_score(best_predicted_values, Y_test.values)


# In[ ]:




