#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys #python3 
import matplotlib.pyplot as plt
import datetime
from scipy import stats


# In[2]:


# source link: https://www.oecd-ilibrary.org/environment/data/oecd-environment-statistics_env-data-en
#Possible question: Is there any difference in energy tax for climate change between Germany, United States, Brazil and China?

df_climate_tax = pd.read_csv('cleaned_datasets/cleaned_tax_climate.csv')
countries_climate_tax = df_climate_tax[(df_climate_tax['Country'] == 'Germany') | 
                                       (df_climate_tax['Country'] == 'United States')]



countries_climate_tax

#this dataset will leave it for future research.

