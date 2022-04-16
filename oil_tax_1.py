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

#python3 oil_tax_1.py datasets/ourworld-oil-consumption-by-region.csv datasets/6countries_oil_price-CAD_cents_per_litre.csv datasets/greenhouseCO2-worldbank.csv

# filename1 = sys.argv[1]
# filename2 = sys.argv[2]
# filename3 = sys.argv[3]


# In[2]:


#PART A

# https://ourworldindata.org/grapher/oil-consumption-by-country 
# ==== We look at countries have most oil consumption in main regions in the world. 
# The sorted result shows the top 10 most consuming oil in 1994 to 2019 countries


consumption_df = pd.read_csv('datasets/ourworld-oil-consumption-by-region.csv')
# consumption_df = pd.read_csv(filename1)
consumption_df = consumption_df.drop(columns = ['Code'])
consumption_df = consumption_df[(consumption_df['Year'] >= 1994) & (consumption_df['Year'] <= 2019)]
sum_df = consumption_df.groupby(consumption_df['Entity'])['Oil Consumption - TWh'].agg(['sum']).reset_index().rename(columns={'sum': 'Oil Consumption Total - TWh','Entity': 'Country'})
sum_df = sum_df.sort_values(by = 'Oil Consumption Total - TWh', axis = 0, ascending = False)

#-------- Put in report:
consumption_times = (2.662660e+05/26) / (3.765799e+04/26) # mean USA / mean Germany = 7.070637599085878
print("Times of USA to Germany total consumption: ", consumption_times)

sum_df = sum_df.reset_index(level = 0)
sum_df = sum_df.drop([0,1,2,3,4,6,7,9,10,12,13]).reset_index(level = 0) #drop the world regoins
sum_df = sum_df.drop(columns = ['level_0'])
sum_df = sum_df.rename(columns = {'index': 'Rank'})
sum_df['Rank'] = sum_df.index + 1

sum_df.head(10)


# In[3]:


# Part B-1:
# oil tax data (monthly) cleaning, filtering:
df_oil_international = pd.read_csv('datasets/6countries_oil_price-CAD_cents_per_litre.csv')
# df_oil_international = pd.read_csv(filename2)
df_oil_international = df_oil_international.drop(columns = ['Financial Situation'])
df_oil_international['Date'] = pd.to_datetime(df_oil_international['Date'])

#Analyse total oil price trend(base price + tax) in developed countries between North America and Europe:
df_oil_international['Date'] = pd.to_datetime(df_oil_international['Date'])
df_totalPrice = df_oil_international[(df_oil_international['Tax Status'] == 'Total')]
df_totalPrice = df_totalPrice[(df_totalPrice['Date'].dt.year <= 2019)]

# transform data needs reset_index() so that'Date' becomes a column:

df_totalPrice_germany = df_totalPrice.groupby(df_totalPrice['Date'])['Germany'].agg(['sum']).reset_index() #"grouping" on each month, so "sum" is just equal to monthly actual price
df_totalPrice_usa = df_totalPrice.groupby(df_totalPrice['Date'])['USA'].agg(['sum']).reset_index()


plt.plot(df_totalPrice_germany['Date'], df_totalPrice_germany['sum'], color = 'orange') 
plt.plot(df_totalPrice_usa['Date'], df_totalPrice_usa['sum'], color = 'black')


plt.xlabel('year')
plt.ylabel('Oil price')
plt.title('Oil price(tax included) trend in 2012 - 2019')
plt.legend(('Germany', 'USA'), loc = 'upper left')

# df_totalPrice_usa

# df_oil_international
# df_totalPrice_germany


# In[4]:


# PART B-2
# ----- Developed countried data from Open Data Canada, Pre-pandanmic data -----
# The statistics above ranks the most oil consumption countries and region. 
# Under a finite range of datasets, we chose Germany(Europe) and USA(North America) to analyse their relationships of oil tax and energy contributed CO2 emission.

# ** Canadian cents per litre **


#1. oil tax data cleaning, filtering:
df_oil_international = pd.read_csv('datasets/6countries_oil_price-CAD_cents_per_litre.csv')
# df_oil_international = pd.read_csv(filename2)
df_oil_international = df_oil_international.drop(columns = ['Financial Situation'])
df_oil_international['Date'] = pd.to_datetime(df_oil_international['Date']) #convert to Date time object

df_tax = df_oil_international[(df_oil_international['Tax Status'] == 'Tax')]
df_tax = df_tax[(df_tax['Date'].dt.year <= 2019)]



# ***** Original not-yet normalized model *****


#2.H0: Are both Germany and the USA data normally-distributed?  We used normal test to determine it. The p-value is 3.849357578096492e-52 for the data of USA oil tax 
# and p-value is 9.906815927253717e-16 for data of Germany oil tax. 
#Obviously, both p-values reject the H0, so they are both not normal distributions.
print("Normal test USA p-val: ",stats.normaltest(df_tax['USA']).pvalue) # 3.849357578096492e-52
print("Normal test Germany p-val: ", stats.normaltest(df_tax['Germany']).pvalue) # 9.906815927253717e-16


#3. Levene equal variance test:
# We use levene equal variance test to decide if they have different variances. 
#Let's assume H0: they have equal variance. 
#The equal variance test failed - they have different variances because p-value < 0.05 which means H0 is rejected.
print("Levene p-value:", stats.levene(df_tax['USA'], df_tax['Germany']).pvalue) #1.1910589753871016e-08


#4. Attempt to transform data to normal:
#the Germany oil tax data seems a left-skewed model, so firstly, we tried squaring data points to shape it.
squared_germany_df = np.square(df_tax['Germany'])
print("Transform squared Germany: ", stats.normaltest(squared_germany_df).pvalue) #T-test checks if transformed data is normalized.
#The square function does some effect on the data points comparing the p-values. 
#p-value is 1.7131236506779964e-10 after square tranformed,  and previous was 9.906815927253717e-16, but model is still not normal enough.

expo_germany_df = np.exp(df_tax['Germany'])
print("Transform exponential Germany: ", stats.normaltest(expo_germany_df).pvalue)
#exponential transformed: 3.8266418995187167e-31, no better than square transformed, even further to 0.05.



#------- Central limit theorem ------
df_yr_mean_usa = df_totalPrice.groupby(df_totalPrice['Date'].dt.year)['USA'].agg(['mean']).reset_index()
df_yr_mean_germany = df_totalPrice.groupby(df_totalPrice['Date'].dt.year)['Germany'].agg(['mean']).reset_index()

usa_pvalue_clt = stats.normaltest(df_yr_mean_usa['mean'])
print("----> USA mean normal test: ", usa_pvalue_clt) #pvalue=0.5323075953228054

germany_pvalue_clt = stats.normaltest(df_yr_mean_germany['mean'])
print("----> Germany mean normal test: ", germany_pvalue_clt) #pvalue=0.7753181387281864

# UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=8
#The p-value passed when using mean values of each year. 
# However, it raises an issue because we cannot really rely on the asymptotic results with only 8 observations.




#5. plot histogram
#----- Histograms shows the Germany as well as EU commitee has much higher treshold for oil and gas tax When the climate change policy was ruled out (ADD Source: ).
#----- USA oil tax has much lower oil tax rate comparing to Germany.


plt.hist(df_tax['USA'], bins = 25, alpha = 0.5, color = 'red')
plt.hist(df_tax['Germany'], bins = 25, alpha = 0.7, color = 'orange') #alpha = 70% opacity

plt.xlabel('CAD cents per litre')
plt.ylabel('Quantity by month') 
plt.title('Germany and USA Oil and Gas Tax 2012 - 2019', fontsize= 16)
plt.legend(('USA', 'Germany'), loc = 'upper left')

plt.savefig('usa_germany_oiltax.png')

#6.Moving forward, we use Mann-whitney U-test to see if the USA oil tax rates are different to the Germany's.
utest, p_value = stats.mannwhitneyu(df_tax['USA'], df_tax['Germany'], alternative = 'two-sided')
#Result: Mann-Whitney test is 0.0 which means that all USA data points are different to all Germany's data points.
# Also, the mean values is 9 times different, so it backs up that their distributions are further apart.

print("Mann-Whitney U-test, p-value: ", utest, p_value) 


# In[5]:


#7. Calculate oil tax of USA, and GERMANY mean:

print('Germany oil tax average of 2012 - 2019: ', df_tax['Germany'].mean()) #127.13437499999998 CAD
print('USA oil tax average of 2012 - 2019: ', df_tax['USA'].mean()) # 14.241666666666662 CAD
print('Oil tax times between two countries: ', df_tax['Germany'].mean() / df_tax['USA'].mean())
#On average, Germany oil tax is 8.9 times higher than the USA's in 2012-2019.


# In[6]:


#PART C
#Since the limited data points in oil tax for looking at the correlation between oil tax and CO2 emission, 
#we'll focus on the trend of Germany and USA CO2 emission from 1975 to 2018 by predicting linear regression:
# -------> Question: Is high oil tax in effect on reducing gas emission(CO2) in Germany? 
# In contrast, does the stable low oil tax keep Gas emission more in USA?
# Data source: World Bank

#1.cleanup, transpose data:

df_climate = pd.read_csv('datasets/greenhouseCO2-worldbank.csv')
# df_climate = pd.read_csv(filename3)

df_climate = df_climate.drop(columns = (["Country Code", 'Series Code']))
# Germany:
df_climate_germany = df_climate[(df_climate['Country Name'] == 'Germany')]
df_climate_germany = df_climate_germany[(df_climate_germany['Series Name'] == 'Total greenhouse gas emissions (kt of CO2 equivalent)')]
df_climate_germany = df_climate_germany.drop(columns = ['Country Name', 'Series Name'])

#USA:
df_climate_usa = df_climate[(df_climate['Country Name'] == 'United States')]
df_climate_usa = df_climate_usa[(df_climate_usa['Series Name'] == 'Total greenhouse gas emissions (kt of CO2 equivalent)')]
df_climate_usa = df_climate_usa.drop(columns = ['Country Name', 'Series Name'])


#2.transpose columns to rows, re-create new dataframe:

#Germany:
df_emission_germany = df_climate_germany.transpose().rename(columns = {9284 : 'Total_CO2_emission(kt)'}) 
df_emission_germany = df_emission_germany.reset_index(level = 0).rename(columns={'index': 'Year'})

#USA:
df_emission_usa = df_climate_usa.transpose().rename(columns = {26175 : 'Total_CO2_emission(kt)'}) 
df_emission_usa = df_emission_usa.reset_index(level = 0).rename(columns={'index': 'Year'})


#3.convert object to integer, float:
#Germany:
df_emission_germany['Year'] = df_emission_germany['Year'].astype(int) 
df_emission_germany = df_emission_germany[(df_emission_germany['Year'] >= 1975) & (df_emission_germany['Year'] <= 2018)]

df_emission_germany['Total_CO2_emission(kt)'] = df_emission_germany['Total_CO2_emission(kt)'].astype(float)


#USA:
df_emission_usa['Year'] = df_emission_usa['Year'].astype(int) 
df_emission_usa = df_emission_usa[(df_emission_usa['Year'] >= 1975) & (df_emission_usa['Year'] <= 2018)]

df_emission_usa['Total_CO2_emission(kt)'] = df_emission_usa['Total_CO2_emission(kt)'].astype(float)

#sort
df_emission_germany = df_emission_germany.sort_values(by = ['Year'], ascending = True)
df_emission_usa = df_emission_usa.sort_values(by = ['Year'], ascending = True)

# concat_emission = pd.concat([df_emission_germany, df_emission_usa], axis = 1).reindex(df_emission_usa.index)


#4.Linear regression - CO2 Emission from 1975 to 2018:
#Germany:
fit_germany_CO2 = stats.linregress(df_emission_germany['Year'], df_emission_germany['Total_CO2_emission(kt)'])
df_emission_germany['prediction'] = df_emission_germany['Year']*fit_germany_CO2.slope + fit_germany_CO2.intercept 

#USA:
fit_usa_CO2 = stats.linregress(df_emission_usa['Year'], df_emission_usa['Total_CO2_emission(kt)'])
df_emission_usa['prediction'] = df_emission_usa['Year']*fit_usa_CO2.slope + fit_usa_CO2.intercept 

print("- Germany CO2 emission: correlation ", fit_germany_CO2.rvalue, "Slope ", fit_germany_CO2.slope)
print("- USA CO2 emission: correlation ", fit_usa_CO2.rvalue, "Slope ", fit_usa_CO2.slope)


                                   
#5.plot

#USA: - upward
plt.figure(figsize=(8, 6), dpi=80)
plt.plot(df_emission_usa['Year'], df_emission_usa['Total_CO2_emission(kt)'], 'b.', alpha = 1) 
plt.plot(df_emission_usa['Year'], df_emission_usa['prediction'], 'r-', linewidth = 1)
plt.title('USA CO2 Emission(kt) Trend from 1975 to 2018')
plt.xlabel('Year')
plt.ylabel('CO2 Emission(kt)')

plt.savefig('usa_co2_emission.png')

# Germany: - downward
plt.figure(figsize=(8, 6), dpi=80)
plt.plot(df_emission_germany['Year'], df_emission_germany['Total_CO2_emission(kt)'], 'b.', alpha = 1) 
plt.plot(df_emission_germany['Year'], df_emission_germany['prediction'], 'r-', linewidth = 1)

plt.savefig('germany_co2_emission.png')

plt.title('Germany CO2 Emission(kt) Trend from 1975 to 2018')
plt.xlabel('Year')
plt.ylabel('CO2 Emission(kt)')


#6.CO2 Emission mean:

emission_times_mean = df_emission_usa['Total_CO2_emission(kt)'].mean() / df_emission_germany['prediction'].mean()
print("CO2 emission times of USA to Germany (mean): ", emission_times_mean)
print("Times of the slope of Germany to the slope of USA: ", fit_germany_CO2.slope / fit_usa_CO2.slope)



# In[7]:


# =============== For Countries and pump prices:

df_tax_5yr = df_tax[(df_tax['Date'].dt.year <= 2016) & (df_tax['Date'].dt.year >= 2012)] 
df_tax_5yr #Units: Candaian Cents per litre

usa_5yr = df_tax_5yr.groupby(df_tax_5yr['Date'].dt.year)['USA'].agg('mean').reset_index()

# linear regression:
fit_usa_5yr = stats.linregress(usa_5yr['Date'], usa_5yr['USA'])
usa_5yr['prediction'] = usa_5yr['Date']*fit_usa_5yr.slope + fit_usa_5yr.intercept 

print("R-value of USA oil tax mean(2012 - 2016): ", fit_usa_5yr.rvalue)
print("Slope of USA oil tax mean(2012 - 2016): ", fit_usa_5yr.slope)

#Plot U.S.A mean oil tax from 2012 to 2016
plt.plot(usa_5yr['Date'], usa_5yr['USA'], 'b.', alpha = 1) 
plt.plot(usa_5yr['Date'], usa_5yr['prediction'], 'r-', linewidth = 1)
plt.title('U.S.A oil tax mean from 2012 to 2016')
plt.xlabel('Year')
plt.ylabel('U.S.A. oil tax mean (Units: Candaian Cents per litre)')


plt.savefig('usa_mean_2016.png')

