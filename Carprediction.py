
# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("data/car data.csv")


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


data.drop(['Car_Name'], axis =1, inplace = True)


# In[6]:


data = pd.get_dummies(data, drop_first=True)


# In[7]:


data['current'] =2020


# In[8]:


data['No_of_Year'] = data['current'] - data['Year']


# In[9]:



data.drop(['current', 'Year'], axis =1, inplace = True)


# In[10]:


data.head()


# In[11]:


plt.figure(figsize=(20,15))
sns.heatmap(data.corr(), annot =True)


# In[12]:


X= data.iloc[:, 1:]
y = data.iloc[:, 0]
X.head()


# In[13]:


from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()


# In[14]:


#####Feature importance
model.fit(X,y)
print(model.feature_importances_)


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train,X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2)


# In[17]:


from sklearn.ensemble import RandomForestRegressor


# In[18]:


regressor =  RandomForestRegressor()


# In[36]:


n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1500, num = 30)]
print(n_estimators)


# In[37]:


from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1500, num = 30)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(5, 50, num = 6)]

min_samples_split = [2, 4, 5, 8, 10, 12, 15, 40, 80, 100]

min_samples_leaf = [1, 2, 5, 10]


# In[38]:



random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
rf = RandomForestRegressor()


# In[39]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[40]:


rf_random.fit(X_train,y_train)
rf_random.score(X_test, y_test)


# In[41]:


predictions = rf_random.predict(X_test)


# In[42]:


plt.scatter(y_test, predictions)


# In[43]:


sns.distplot( y_test - predictions)


# In[44]:


file = open("RandomForestRegressor.pkl", 'wb')
pickle.dump(rf_random, file)


# In[45]:


data.columns


# In[46]:



rf_random


# In[47]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[48]:


rf_random.best_params_


# In[ ]:




