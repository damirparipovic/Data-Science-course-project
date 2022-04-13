#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys #python3 
import matplotlib.pyplot as plt
import datetime
from scipy import stats
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

# Citation: https://stackoverflow.com/questions/50594613/how-to-plot-aggregated-by-date-pandas-dataframe


# In[2]:


# ----- Developed countried data in Organization of Economic Cooperation and Development (OECD) -----
# ----- Pre-pandanmic data -------
# ** Canadian cents per litre **

#1. oil tax data cleaning, filtering:
df_oil_international = pd.read_csv('datasets/6countries_oil_price-CAD_cents_per_litre.csv')
df_oil_international = df_oil_international.drop(columns = ['Financial Situation'])
df_oil_international['Date'] = pd.to_datetime(df_oil_international['Date']) #convert to Date time object

df_tax = df_oil_international[(df_oil_international['Tax Status'] == 'Tax')]
df_tax = df_tax[(df_tax['Date'].dt.year <= 2019)]


#2. ***** Original not-yet normalized model *****

#----- Histograms shows the Germany as well as EU commitee has much higher treshold for oil and gas tax When the climate change policy was ruled out (ADD Source: ).
#----- USA oil tax has much lower oil tax rate comparing to Germany. And Japan oil tax rates r ate in the middle of these two countries'.
# plot histogram
plt.figure(figsize = (10, 5))

plt.hist(df_tax['USA'], bins = 25, alpha = 0.7, color = 'blue')
plt.hist(df_tax['Germany'], bins = 25, alpha = 0.7, color = 'orange') #alpha = 70% opacity

plt.xlabel('CAD cents per litre')
plt.ylabel('Quantity by month') 
plt.title('Germany and USA Gas Tax 2012 - 2019', fontsize= 16)
plt.legend(('USA', 'Germany'), loc = 'upper left')


#3.H0: Are both Germany and the USA data normally-distributed?  We used T-test to determine it. The p-value is 3.849357578096492e-52 for the data of USA oil tax 
# and p-value is 9.906815927253717e-16 for data of Germany oil tax. 
#Obviously, both p-values reject the H0, so they are both not normal distributions.
print("T-test USA p-val: ",stats.normaltest(df_tax['USA']).pvalue) # 3.849357578096492e-52
print("T-test Germany p-val: ", stats.normaltest(df_tax['Germany']).pvalue) # 9.906815927253717e-16


#4. Levene equal variance test:
# From the plot of original data, it looked like Germnay tax rates are more spread out than USA's. 
# We use levene equal variance test to decide if they have different variances. 
#Let's assume H0: they have equal variance. 
#The result is true to what the plot looked like - they have different variances because p-value < 0.05 which means H0 is rejected.
print("Leven p-value:", stats.levene(df_tax['USA'], df_tax['Germany']).pvalue) #1.1910589753871016e-08


#5. Transforming data to normal-enough
#the Germany oil tax data seems a left-skewed model, so firstly, we tried squaring data points to shape it.
squared_Germany_df = np.square(df_tax['Germany'])
print("Transformed Germany: ", stats.normaltest(squared_Germany_df).pvalue) #T-test checks if transformed data is normalized.
#The square function does some effect on the data points comparing the p-values. 
#Right now is 1.7131236506779964e-10 to previous was 9.906815927253717e-16, but model is still not normal enough.

#Secondly, we  Central limit theorom OR Mann-whitney 


# In[3]:



# oil tax data cleaning, filtering:
df_oil_international = pd.read_csv('datasets/6countries_oil_price-CAD_cents_per_litre.csv')
df_oil_international = df_oil_international.drop(columns = ['Financial Situation'])
df_oil_international['Date'] = pd.to_datetime(df_oil_international['Date'])

#Analyse total oil price trend(base price + tax) in developed countries between North America and Europe:
df_oil_international['Date'] = pd.to_datetime(df_oil_international['Date'])
df_totalPrice = df_oil_international[(df_oil_international['Tax Status'] == 'Total')]
df_totalPrice = df_totalPrice[(df_totalPrice['Date'].dt.year <= 2019)]

# transform data needs reset_index() so that'Date' becomes a column:
df_totalPrice_germany = df_totalPrice.groupby(df_totalPrice['Date'])['Germany'].agg(['mean']).reset_index()
df_totalPrice_canada = df_totalPrice.groupby(df_totalPrice['Date'])['Canada'].agg(['mean']).reset_index()
df_totalPrice_france = df_totalPrice.groupby(df_totalPrice['Date'])['France'].agg(['mean']).reset_index()
df_totalPrice_usa = df_totalPrice.groupby(df_totalPrice['Date'])['USA'].agg(['mean']).reset_index()
# df_totalPrice_usa = df_totalPrice.groupby(df_totalPrice['Date'].dt.year)['USA'].agg(['mean']).reset_index()

plt.plot(df_totalPrice_germany['Date'], df_totalPrice_germany['mean'], color = 'orange') 
plt.plot(df_totalPrice_france['Date'], df_totalPrice_france['mean'], color = 'green')
plt.plot(df_totalPrice_canada['Date'], df_totalPrice_canada['mean'], color = 'blue')
plt.plot(df_totalPrice_usa['Date'], df_totalPrice_usa['mean'], color = 'black')


plt.xlabel('year')
plt.ylabel('Oil price')
plt.title('Oil price(tax included) trend in 2012 - 2019')
plt.legend(('Germany', 'France', 'Canada', 'USA'), loc = 'upper left')

df_totalPrice_usa


# In[6]:


# -------> Question: Is high oil tax in effect on reducing Gas emission(CO2) in Germany?

#1. ----------- cleanup, transpose data
df_climate = pd.read_csv('datasets/climatechange1.csv')
df_climate = df_climate.drop(columns = (["Country Code", 'Series Code']))
# df_climate = df_climate.style.set_properties(subset = ['Series Name'], **{'width': '300px'})
df_climate_germany = df_climate[(df_climate['Country Name'] == 'Germany')]
df_climate_germany = df_climate_germany[(df_climate_germany['Series Name'] == 'Total greenhouse gas emissions (kt of CO2 equivalent)')]
df_climate_germany = df_climate_germany.drop(columns = ['Country Name', 'Series Name'])

#2.transpose columns to rows, re-create new dataframe:
df_emission_germany = df_climate_germany.transpose().rename(columns = {9284 : 'CO2_emission(kt)'}) 
df_emission_germany = df_emission_germany.reset_index(level = 0).rename(columns={'index': 'Year'})


#convert object to integer, float:
df_emission_germany['Year'] = df_emission_germany['Year'].astype(int) 
df_emission_germany = df_emission_germany[(df_emission_germany['Year'] >= 1975) & (df_emission_germany['Year'] <= 2018)]

df_emission_germany['CO2_emission(kt)'] = df_emission_germany['CO2_emission(kt)'].astype(float)

#display
df_emission_germany = df_emission_germany.sort_values(by = ['Year'], ascending = True)

#3.Linear regression:
fit_Germany_CO2 = stats.linregress(df_emission_germany['Year'], df_emission_germany['CO2_emission(kt)'])
df_emission_germany['prediction'] = df_emission_germany['Year']*fit_Germany_CO2.slope + fit_Germany_CO2.intercept 


#4.plot
plt.figure(figsize=(12, 8), dpi=80)
plt.plot(df_emission_germany['Year'], df_emission_germany['CO2_emission(kt)'], color = 'orange') 

plt.plot(df_emission_germany['Year'], df_emission_germany['prediction'], 'r-', linewidth = 2)

# df_emission_germany


# In[ ]:




