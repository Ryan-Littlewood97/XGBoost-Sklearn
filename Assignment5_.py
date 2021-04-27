#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports 
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
from sklearn import metrics


# In[2]:


#Load data, Split into X and Y 
diabetes = load_diabetes()
X = diabetes['data']
y = diabetes['target']


# In[3]:


#Convert X (Main data) into pandas DF 
diabetes_df = pd.DataFrame(X, columns=diabetes['feature_names'])
diabetes_df.head()


# In[4]:


#80/20 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# In[5]:


#Create the model, and tune the hperparameters using randomised grid search
param_grid = [{'max_depth': list(np.arange(2,6,1)), 'learning_rate': list(np.arange(.2,1,.1)),
              'n_estimators' : list(np.arange(2,101,10)), 'reg_alpha': list(np.arange(.1,1.1,1)), 
               'reg_lambda': list(np.arange(0.1,1.1,1))}]
model = xgb.XGBRegressor()


# In[6]:


xg = RandomizedSearchCV(model,param_distributions = param_grid, n_iter = 1000, 
                         cv=5, scoring = 'neg_mean_squared_error')


# In[7]:


#Test the model using the test split of the data 
xg.fit(X_train, y_train)


# In[8]:


#Train the final model on the Test data 
y_pred  = xg.predict(X_test)


# In[9]:


#These are the Model printouts for the test model (xg): Best parameters, MSE, and MAE 
print(xg.best_params_)
print('MSE: {}\nMAE: {}'.format(metrics.mean_squared_error(y_test,y_pred), metrics.mean_absolute_error(y_test,y_pred)))


# In[10]:


#Train the final Regressor model using the best parameters from the tester model (xg)
fmodel = xgb.XGBRegressor(**xg.best_params_)
fmodel.fit(X_train, y_train)
#
y_pred  = fmodel.predict(X_test)


# In[11]:


#THese are the printouts for the final model, using the dictionary of best params 
# produced by the tester model (xg): MSE, and MAE 
print('MSE: {}\nMAE: {}'.format(metrics.mean_squared_error(y_test,y_pred), metrics.mean_absolute_error(y_test,y_pred)))


# In[12]:


xgb.plot_importance(fmodel)
plt.show()


# In[ ]:




