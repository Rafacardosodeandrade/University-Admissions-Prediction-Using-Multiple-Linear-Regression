# Task #1: Understand the Problem Statement

# In this project, we will buil a regression model 
# to predict the chance of admission into a particular 
# university based on the student's profile.

# => INPUTS (FEATURES):
# - GRE Scores (out of 340)
# - TOEFL Scores (out of 120)
# - University Rating (out of 5)
# - Statement of Purpose (SOP)
# - Letter of Recommendation (LOR) Strengh (out of 5)
# - Undergraduate GPA (out of 10)
# - Research Experience (either 0 or 1)

# OUTPUTS:
# - Chance of admission (ranging from 0 to 1)

###########################################################

#Task 2 - Import Libraries and Dataset

import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from jupyterthemes import jtplot
jtplot.style(theme='monokai' , context='notebook' , ticks=True, grid=False)

# read the csv file

admission_df = pd.read_csv('Admission_Predict.csv')
admission_df.head()

# Let's drop the serial no.
admission_df.drop('Serial No.' , axis = 1, inplace = True)
admission_df


#############################################################
#Task3: Perform Exploratory Data Analysis

# checking the null values
admission_df.isnull().sum()

# check the dataframe information
admission_df.info()

# statistical summary of the dataframe
admission_df.describe()

# grouping by University ranking
df_university = admission_df.groupby(by = 'University Rating').mean()
df_university

#############################################################
#Task4: Perform Data Visualization

#matplotlib visualizations
admission_df.hist(bins = 30, figsize = (20, 20), color = 'r') 

#pairplot using seaborn
sns.pairplot(admission_df) 

#plot correlations heatmap - VERY INTERSTING
corr_matrix = admission_df.corr()
plt.figure(figsize = (12, 12))
sns.heatmap(corr_matrix, annot = True)
plt.show()

#Task #5: Create Training and Testing Dataset
admission_df.columns
x = admission_df.drop(columns = ['Chance of Admit'])
y = admission_df['Chance of Admit']
x.shape
y.shape
y 
x = np.array(x)
y = np.array(y)
y = y.reshape(-1,1)
y.shape

#scaling the data before training the model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

scaler_y = StandardScaler()
y = scaler_x.fit_transform(y)

#spliting the data in to test and train sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.15)

# Task#6: Train and evaluate a linear regression model
from sklear.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

LinearRegression_model = LinearRegression()
LinearRegression_model.fit(x_train, y_train)

accuracy_LinearRegression = LinearRegression_model.score(x_test, y_test)
accuracy_LinearRegression

#Task 7: Train and Evaluate an Artificial Neural Network

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

ANN_model = keras.Sequential()
ANN_model.add(Dense(50, input_dim = 7))
ANN_model.add(Activation('relu'))
ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))
ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))
ANN_model.add(Dense(50))
ANN_model.add(Activation('linear'))
ANN_model.add(Dense(1))
ANN_model.compile(loss = 'mse', optimizer = 'adam')
ANN_model.summary()

ANN_model.compile(optimizer = 'Adam' , loss = 'mean_squared_error')
epochs_hist = ANN_model.fit(x_train, y_train, epochs = 100, batch_size = 20)

epochs_hist.history.keys()

plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])

#Task#8: Train and Evaluate a decision tree and random forest models

from sklearn.tree import DecisionTreeRegressor
DecisionTree_model = DecisionTreeRegressor()
DecisionTree_model.fit(x_train, y_train)

accuracy_DecisionTree = DecisionTree_model.score(x_test, y_test)
accuracy_DecisionTree

from sklearn.ensemble import RandomForestRegressor
RandomForest_model = RandomForestRegressor(n_estimators=100, max_depth = 10)
RandomForest_model.fit(x_train, y_train)

accuracy_RandomForest = RandomForest_model.score(x_test, y_test)
accuracy_RandomForest

#Task9 - KPIS explain

#Task#10 - Calculate Regression Model KPIS.
y_predict = LinearRegression_model.predict(x_test)
plt.plot(y_test, y_predict, '^', color = 'r')

y_predict_orig = scaler_y.inverse_transform(y_predict)
y_test_orig = scaler_y.inverse_transform(y_test)

plt.plot(y_test_orig, y_predict_orig, '^', color = 'r')

k = x_test.shape[1]
n = len(x_test)
n

from sklear.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =', MSE, '\nMAE =', MAE, '\nR2 =', r2, 'nAdjusted R2 =' , adj_r2)


