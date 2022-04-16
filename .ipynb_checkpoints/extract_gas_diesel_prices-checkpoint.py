# imports
import numpy as np
import pandas as pd
import re
import sys
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans

# functions
def extractSeries(incoming_data, year_start, year_end, series_name, column_name):
    data = incoming_data.copy(deep=True)
    data.columns = data.columns.str.rstrip("[YR1234567890]")
    data.columns = data.columns.str.rstrip(" ")
    
    data = data.replace("..", np.nan)
    # Either of the two below work
    #gasoline_prices = data[data['Series Name'] == data["Series Name"][0]]
    gasoline_prices = data[data['Series Name'] == series_name]

    years = [str(x) for x in range(year_start, year_end + 1)]
    #years = ['1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000']

    gasoline_prices = gasoline_prices.drop('Series Name', axis=1)

    # The below is what i want but the countries are not staying grouped together, need to sort
    melted_gas_prices = pd.melt(gasoline_prices, id_vars=["Country Name"], value_vars=years, var_name='year', value_name=column_name)
    #melted_gas_prices.set_index("Country Name")

    sorted_gas_prices = melted_gas_prices.sort_values(by=["Country Name", "year"])
    sorted_gas_prices = sorted_gas_prices.astype({column_name:"float"})
    return sorted_gas_prices

def getCountriesWithValues(incoming_df, column_to_check):
    # to column_to_check pass either "gas prices" or "diesel prices"
    single_year_df_2014 = incoming_df[incoming_df["year"] == "2014"]
    na_removed_2014 = single_year_df_2014[single_year_df_2014[column_to_check].notna()]
    
    single_year_df_2016 = incoming_df[incoming_df["year"] == "2016"]
    na_removed_2016 = single_year_df_2016[single_year_df_2016[column_to_check].notna()]
    
    countries_2014 = list(na_removed_2014["Country Name"].unique())
    countries_2016 = list(na_removed_2016["Country Name"].unique())
    
    countries = list(set(countries_2014).intersection(countries_2016))
    countries.sort()
    
    return countries

def getFilteredPrices(incoming_dataframe, col):
    #pass dataframe for each country
    new_dataframe = incoming_dataframe.filter(items=['year', col])
    
    lowess_smoothed = sm.nonparametric.lowess(
        new_dataframe[col],
        new_dataframe['year'],
        frac=0.2)
    
    lowess_x = list(zip(*lowess_smoothed))[0]
    lowess_y = list(zip(*lowess_smoothed))[1]
    
    f = interp1d(lowess_x, lowess_y, bounds_error=False)
    
    name = "filtered " + col
    new_dataframe[name] = new_dataframe['year'].apply(lambda x: f(x))
    
    return new_dataframe

def filterEachCountry(d4, func_countries_list, col):
    filtered_countries = pd.DataFrame()
    for country in func_countries_list:
        country_df = d4[d4["Country Name"] == country]
        #print(country_df)
        country_df_w_filtered_prices = pd.concat([country_df.filter(items=["Country Name"]), getFilteredPrices(country_df, col)], axis=1)
        #print(country_df_w_filtered_prices)
        filtered_countries = pd.concat([filtered_countries, country_df_w_filtered_prices], axis=0)

    return filtered_countries

def main():
    
    data = pd.read_csv("./datasets/WDI_data_oil_price_and_other_related.csv")
    data = data.drop(["Series Code", "Country Code"], axis=1)
    
    # get gasoline
    raw_gasoline = extractSeries(data, 1991, 2016, "Pump price for gasoline (US$ per liter)", "gas prices")
    
    # get all countries with valid entries
    countries_list_gasoline = getCountriesWithValues(raw_gasoline, "gas prices")
    
    # lowess filter all of the gasoline prices for each country
    final_gasoline_df = filterEachCountry(raw_gasoline, countries_list_gasoline, "gas prices").set_index("Country Name")
    final_gasoline_df.to_csv("./datasets/gasoline-prices-by-country.csv")
    
    #get diesel
    raw_diesel = extractSeries(data, 1991, 2016, "Pump price for diesel fuel (US$ per liter)", "diesel prices")
    #print(raw_diesel)
    countries_list_diesel = getCountriesWithValues(raw_diesel, "diesel prices")
    final_diesel_df = filterEachCountry(raw_diesel, countries_list_diesel, "diesel prices").set_index("Country Name")
    final_diesel_df.to_csv("./datasets/diesel-prices-by-country.csv")
    
    return

if __name__ == "__main__":
    main()