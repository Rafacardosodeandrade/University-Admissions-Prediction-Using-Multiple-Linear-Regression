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