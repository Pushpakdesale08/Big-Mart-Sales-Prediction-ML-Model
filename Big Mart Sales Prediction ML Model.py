#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df_test = pd.read_csv('test_AbJTz2l.csv')
df_train = pd.read_csv('train_v9rqX0R.csv')


# In[3]:


df_train.head()


# In[4]:


df_train = df_train.drop(columns =['Outlet_Identifier'])
df_train


# In[5]:


df_train = df_train.drop(columns =['Item_Identifier'])
df_train


# In[6]:


df_train.dtypes


# In[7]:


df_train.shape


# In[8]:


df_train.columns


# ## Find Null Values

# In[9]:


df_train.isnull().sum()


# In[10]:


df_test.isnull().sum()


# In[11]:


df_train.info()


# ## EDA

# In[12]:


df_train.describe()


# In[13]:


df_train['Item_Weight'].describe()


# In[14]:


df_train['Item_Weight'].fillna(df_train['Item_Weight'].mean(),inplace=True)
df_test['Item_Weight'].fillna(df_test['Item_Weight'].mean(),inplace=True)


# In[15]:


df_train.isnull().sum()


# In[16]:


df_train['Item_Weight'].describe()


# In[17]:


df_train['Outlet_Size']


# In[18]:


df_train['Outlet_Size'].value_counts()


# In[19]:


df_train['Outlet_Size'].mode()


# In[20]:


df_train['Outlet_Size'].fillna(df_train['Outlet_Size'].mode()[0],inplace=True)
df_test['Outlet_Size'].fillna(df_test['Outlet_Size'].mode()[0],inplace=True)


# In[21]:


df_train.isnull().sum()


# In[22]:


df_test.isnull().sum()


# ## Label Encoding

# In[23]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df_train['Item_Fat_Content']= le.fit_transform(df_train['Item_Fat_Content'])
df_train['Item_Type']= le.fit_transform(df_train['Item_Type'])
df_train['Outlet_Size']= le.fit_transform(df_train['Outlet_Size'])
df_train['Outlet_Location_Type']= le.fit_transform(df_train['Outlet_Location_Type'])
df_train['Outlet_Type']= le.fit_transform(df_train['Outlet_Type'])


# ## Train Test_Split Data

# In[24]:


df_train


# In[25]:


X=df_train.drop('Item_Outlet_Sales',axis=1)

y=df_train['Item_Outlet_Sales']


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=101, test_size=0.2)


# ## Standarization

# In[27]:


X.describe()


# In[28]:


from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
sc= StandardScaler()


# In[29]:


X_train_std= sc.fit_transform(X_train)
X_test_std= sc.transform(X_test)


# In[30]:


X_train_std


# In[31]:


X_test_std


# In[32]:


y_train


# In[33]:


y_test


# ## Model Building

# In[34]:


X_test.head()


# In[35]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ## Linear Regression

# In[36]:


from sklearn.linear_model import LinearRegression
lr= LinearRegression()


# In[37]:


lr.fit(X_train_std,y_train)


# In[38]:


y_pred_lr=lr.predict(X_test_std)


# In[39]:


print(r2_score(y_test,y_pred_lr))
print(mean_absolute_error(y_test,y_pred_lr))
print(np.sqrt(mean_squared_error(y_test,y_pred_lr)))


# ## Random Forest Regressor

# In[40]:


from sklearn.ensemble import RandomForestRegressor

rf= RandomForestRegressor(n_estimators=1000)


# In[41]:


rf.fit(X_train_std,y_train)


# In[42]:


y_pred_rf= rf.predict(X_test_std)


# In[43]:


print(r2_score(y_test,y_pred_rf))
print(mean_absolute_error(y_test,y_pred_rf))
print(np.sqrt(mean_squared_error(y_test,y_pred_rf)))


# ## XG Boost Regressor

# In[44]:


from xgboost import XGBRegressor
xg= XGBRegressor()


# In[45]:


xg.fit(X_train_std, y_train)


# In[46]:


y_pred_xg=xg.predict(X_test_std)


# In[47]:


print(r2_score(y_test,y_pred_xg))
print(mean_absolute_error(y_test,y_pred_xg))
print(np.sqrt(mean_squared_error(y_test,y_pred_xg)))


# ## Hyper parameter tuning

# In[48]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


# In[50]:


model = RandomForestRegressor()
n_estimators = [10, 100, 1000]
max_depth=range(1,31)
min_samples_leaf=np.linspace(0.1, 1.0)
max_features=["auto", "sqrt", "log2"]
min_samples_split=np.linspace(0.1, 1.0, 10)

grid = dict(n_estimators=n_estimators)

grid_search_forest = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1,
scoring='r2',error_score=0,verbose=2,cv=2)
grid_search_forest.fit(X_train_std, y_train)

print(f"Best: {grid_search_forest.best_score_:.3f} using {grid_search_forest.best_params_}")
means = grid_search_forest.cv_results_['mean_test_score']
stds = grid_search_forest.cv_results_['std_test_score']
params = grid_search_forest.cv_results_['params']


# In[51]:


for mean, stdev, param in zip(means, stds, params):
    print(f"{mean:.3f} ({stdev:.3f}) with: {param}")


# In[53]:


grid_search_forest.best_params_


# In[52]:


grid_search_forest.best_score_


# In[55]:


y_pred_rf_grid=grid_search_forest.predict(X_test_std)


# In[57]:


r2_score(y_test,y_pred_rf_grid)

