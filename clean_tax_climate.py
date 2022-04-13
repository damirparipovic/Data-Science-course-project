# ============ Energy(oil and gas) tax and climate change data cleanup ==========


#//open spark on Linux:
#PATH=/usr/shared/CMPT/big-data/spark-3.1.2-bin-hadoop3.2/bin/:${PATH}
#export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
#export PYSPARK_PYTHON=python3
#spark-submit clean_tax_climate.py datasets/environmental_tax_OECD.csv cleaned_datasets/cleaned_tax_climate.csv
#cd sfuhome/CMPT353/group_project/

#//open shell:
#pyspark

import sys
from pyspark.sql import SparkSession, functions, types, Row
import string, re
import pandas as pd

spark = SparkSession.builder.appName('oil tax and environment').getOrCreate() #Create DF from RDD
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+


#1. create schema
taxes_schema = types.StructType([

    types.StructField('COU', types.StringType()),
    types.StructField('Country', types.StringType()),
    types.StructField('VAR', types.StringType()),
    types.StructField('Variable', types.StringType()),
    types.StructField('CAT', types.StringType()),
    types.StructField('Category', types.StringType()),
    types.StructField('DOM', types.StringType()),
    types.StructField('Environmental domain', types.StringType()),
    types.StructField('YEA', types.IntegerType()),
    types.StructField('Year', types.IntegerType()),
    types.StructField('Unit Code', types.StringType()),
    types.StructField('Unit', types.StringType()),
    types.StructField('PowerCode Code', types.IntegerType()),
    types.StructField('PowerCode', types.StringType()),
    types.StructField('Reference Period Code', types.IntegerType()),
    types.StructField('Reference Period', types.LongType()),
    types.StructField('Value', types.FloatType()),
    types.StructField('Flag Codes', types.IntegerType()),
    types.StructField('Flags', types.StringType())
    
])


def main(in_directory, out_directory):

    #2. import original dataset
    tax_data = spark.read.csv(in_directory, schema = taxes_schema)
    # tax_data.printSchema()


    #3. select necessary columns - filter(remove) null rows, select 'climate change' rows
    tax_data = tax_data.select(
        tax_data['Country'],
        tax_data['Variable'],
        tax_data['Category'],
        tax_data['Environmental domain'],
        tax_data['Year'],
        tax_data['Unit'],
        tax_data['value']
    ).filter(tax_data['value'].isNotNull()).where((tax_data['Environmental domain'] == 'Total environment' 
       ) | (tax_data['Environmental domain'] == 'Climate change'))

    #select all columns, filter out not US Dollar or Percentage rows
    tax_data = tax_data.select('*').filter((tax_data['Unit'] == 'US Dollar')
     | (tax_data['Unit'] == 'Percentage')).where((tax_data['Variable'] == 'Tax revenue, millions USD'))

    #filter category 
    # tax_data = tax_data.select('*').filter((tax_data['Category'] == 'Energy') 
    #     | (tax_data['Category'] == 'All tax bases'))
    

    #4.sort Country, Variable, Environmental domain, year
    tax_data = tax_data.sort(tax_data['Country'].asc(), 
    tax_data['Variable'].asc(), 
    tax_data['Environmental domain'].desc(), 
    tax_data['Year'].asc(),
    tax_data['Category'].desc()
    )

    # tax_data.show(350, False) #false to turn off trucated text in columns

    tax_data.toPandas().to_csv(out_directory, index = False) #turn off index column

if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
