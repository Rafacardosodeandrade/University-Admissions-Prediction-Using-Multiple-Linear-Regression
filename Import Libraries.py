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


import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from jupyterthemes import jtplot
jtplot.style(theme='monokai' , context='notebook' , ticks=True, grid=False)

