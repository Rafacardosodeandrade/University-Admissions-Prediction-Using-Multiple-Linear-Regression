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

admission_df.hist(bins = 30, figsize = (20, 20), color = 'r') #matplotlib visualizations

sns.pairplot(admission_df) #pairplot using seaborn
