#!/usr/bin/env python
# coding: utf-8

# In[375]:


print("test")


# In[545]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")


# In[546]:


df = pd.read_csv('D:laptopPrice.csv')


# In[547]:


df.head(5)


# In[548]:


#The shape function displays the number of rows and columns of the dataset
print(df.shape)


# In[549]:


#Checking for null values in each column and displaying the sum of all null values in each column
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)


# In[550]:


#Removing the rows with empty values
print(df.dropna())


# In[551]:


df.duplicated().sum()


# In[552]:


df = df.drop_duplicates()


# In[553]:


# Display basic information about the dataset
df.info()


# In[554]:


#Checking the data types to see if all the data is in correct format.
df.dtypes


# In[555]:


# count the number of unique values in each column of a DataFrame.
df.nunique()


# In[556]:


#For numerical columns only
df.describe()


# In[557]:


df.duplicated().sum()


# In[558]:


numeric_features = [feature for feature in df.columns if df[feature].dtype != 'object']
cat_features = [feature for feature in df.columns if df[feature].dtype == 'object']
print(" Numerical features: ", numeric_features)
print("Categorical featues:", cat_features)


# In[559]:


df.describe(include = 'object')


# In[560]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")


# In[561]:


df.loc[df['Price'] == 1, 'Price'] = 500


# In[562]:


df.describe()


# In[563]:


df['Price'].describe()


# In[564]:


df.describe(include = 'object')#summary statistics for categorical values


# In[565]:


numeric_features = [feature for feature in df.columns if df[feature].dtype != 'object']
cat_features = [feature for feature in df.columns if df[feature].dtype == 'object']
print("Numerical features: ", numeric_features)
print("Categorical featues:", cat_features)


# In[566]:


import seaborn as sns
plt.figure(figsize=(10,6))
sns.regplot(x="Number of Ratings", y="Price", data=df)


# In[567]:


from scipy import stats
pearson_coef, p_value = stats.pearsonr(df['Number of Ratings'], df['Price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[568]:


plt.figure(figsize=(10,6))
sns.regplot(x="Number of Reviews", y="Price", data=df)


# In[569]:


sns.boxplot(x="brand", y="Price", data=df)


# In[570]:


plt.figure(figsize=(10,6))
sns.boxplot(x="processor_brand", y="Price", data=df)


# In[571]:


plt.figure(figsize=(10,6))
sns.boxplot(x="processor_name", y="Price", data=df)


# In[572]:


plt.figure(figsize=(10,6))
sns.boxplot(x="processor_gnrtn", y="Price", data=df)


# In[573]:


plt.figure(figsize=(10,6))
sns.boxplot(x="ram_gb", y="Price", data=df)


# In[574]:


plt.figure(figsize=(10,6))
sns.boxplot(x="ram_type", y="Price", data=df)


# In[575]:


plt.figure(figsize=(10,6))
sns.boxplot(x="ssd", y="Price", data=df)


# In[576]:


plt.figure(figsize=(10,6))
sns.boxplot(x="hdd", y="Price", data=df)


# In[577]:


plt.figure(figsize=(10,6))
sns.boxplot(x="os", y="Price", data=df)


# In[578]:


plt.figure(figsize=(10,6))
sns.boxplot(x="os_bit", y="Price", data=df)


# In[579]:


plt.figure(figsize=(10,6))
sns.boxplot(x="graphic_card_gb", y="Price", data=df)


# In[580]:


plt.figure(figsize=(10,6))
sns.boxplot(x="weight", y="Price", data=df)


# In[581]:


plt.figure(figsize=(10,6))
sns.boxplot(x="warranty", y="Price", data=df)


# In[582]:


plt.figure(figsize=(10,6))
sns.boxplot(x="Touchscreen", y="Price", data=df)


# In[583]:


plt.figure(figsize=(10,6))
sns.boxplot(x="msoffice", y="Price", data=df)


# In[584]:


plt.figure(figsize=(10,6))
sns.boxplot(x="os_bit", y="Price", data=df)


# In[585]:


plt.figure(figsize=(10,6))
sns.boxplot(x="rating", y="Price", data=df)


# In[586]:


df.drop(['weight', 'warranty', 'Touchscreen','processor_brand','os_bit'], axis = 1, inplace = True)


# In[587]:


df


# In[588]:


df


# In[589]:


#brand', 'processor_name', 'processor_gnrtn', 'ram_gb', 'ram_type', 'ssd', 'hdd', 'os', 
#'graphic_card_gb', 'msoffice', 'rating'
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
df.brand = labelencoder.fit_transform(df.brand)
df.processor_name = labelencoder.fit_transform(df.processor_name)
df.processor_gnrtn = labelencoder.fit_transform(df.processor_gnrtn)
df.ram_gb = labelencoder.fit_transform(df.ram_gb)
df.ram_type = labelencoder.fit_transform(df.ram_type)
df.ssd = labelencoder.fit_transform(df.ssd)
df.hdd = labelencoder.fit_transform(df.hdd)
df.os = labelencoder.fit_transform(df.os)
df.graphic_card_gb = labelencoder.fit_transform(df.graphic_card_gb)
df.msoffice = labelencoder.fit_transform(df.msoffice)
df.rating = labelencoder.fit_transform(df.rating)
from sklearn.preprocessing import LabelEncoder


# In[590]:


df


# In[591]:


x_train=df.iloc[:,0:13]
y_train=df.iloc[:,10]


# In[592]:


x_train.head()


# In[593]:


y_train.head()


# In[594]:


# importing train_test_split from sklearn
from sklearn.model_selection import train_test_split
# splitting the data  # 30% for testing is used
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size = 0.3, random_state = 0)


# In[595]:


#Multiple Linear Regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model_mlr = model.fit(X_train,Y_train)


# In[596]:


#Making price prediction using the testing set (Fit to MLR)
Y_pred_MLR = model_mlr.predict(X_test)


# In[597]:


#Calculating the Mean Square Error for MLR model
mse_MLR = mean_squared_error(Y_test, Y_pred_MLR)
print('The mean square error for Multiple Linear Regression: ', mse_MLR)


# In[ ]:


#The mean square error for Multiple Linear Regression:  0.3674647167443785


# In[598]:


#Calculating the Mean Absolute Error for MLR model
mae_MLR= mean_absolute_error(Y_test, Y_pred_MLR)
print('The mean absolute error for Multiple Linear Regression: ', mae_MLR)


# In[599]:


#Calling the random forest model and fitting the training data
rfModel = RandomForestRegressor()
model_rf = rfModel.fit(X_train,Y_train)


# In[600]:


#Prediction of Laptop prices using the testing data
Y_pred_RF = model_rf.predict(X_test)


# In[601]:


#Calculating the Mean Square Error for Random Forest Model
mse_RF = mean_squared_error(Y_test, Y_pred_RF)
print('The mean square error of price and predicted value is: ', mse_RF)


# In[602]:


#Calculating the Mean Absolute Error for Random Forest Model
mae_RF= mean_absolute_error(Y_test, Y_pred_RF)
print('The mean absolute error of price and predicted value is: ', mae_RF)


# In[603]:


#LASSO Model
#Calling the model and fitting the training data
LassoModel = Lasso()
model_lm = LassoModel.fit(X_train,Y_train)


# In[604]:


#Price prediction uisng testing data
Y_pred_lasso = model_lm.predict(X_test)


# In[605]:


#Mean Absolute Error for LASSO Model
mae_lasso= mean_absolute_error(Y_test, Y_pred_lasso)
print('The mean absolute error of price and predicted value is: ', mae_lasso)


# In[606]:


#Mean Squared Error for the LASSO Model
mse_lasso = mean_squared_error(Y_test, Y_pred_lasso)
print('The mean square error of price and predicted value is: ', mse_lasso)


# In[607]:


scores = [('MLR', mae_MLR),
          ('Random Forest', mae_RF),
          ('LASSO', mae_lasso)
         ]


# In[608]:


mae = pd.DataFrame(data = scores, columns=['Model', 'MAE Score'])
mae


# In[609]:


mae.sort_values(by=(['MAE Score']), ascending=False, inplace=True)

f, axe = plt.subplots(1,1, figsize=(10,7))
sns.barplot(x = mae['Model'], y=mae['MAE Score'], ax = axe)
axe.set_xlabel('Model', size=20)
axe.set_ylabel('Mean Absolute Error', size=20)

plt.show()


# In[ ]:




