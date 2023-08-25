# Oil prices and taxes project
The project was to collect data about oil prices and tax rates in different countries to search for correlations and extract insights from the data. 

## Setup:

- To execute the programs and view the Jupyter notebook you will need the following installed on your machine: Python 3.5+, Jupyter Notebook, and PySpark.
- The following python libraries are required to run the python files and jupyter notebooks: numpy, pandas, matplotlib, datetime, seaborn, scipy, scikit-learn libraries. (See syntax in Libraries section)

- There are multiple coding environments able to run .ipynb files. You can choose such as Anaconda or Visual Studio.


### Libraries:

```
import numpy as np
import pandas as pd
import sys #python3 
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from scipy import stats
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
```

### Running files:

Files ending with .ipynb can run in a Jupyter Notebook environment directly. Files ending with .py should be executed in a terminal.


### Running Spark jobs
The PySpark job should be ran in Linux-like environment. First, open a terminal(command-prompt), then copy and paste these lines:

```
PATH=/usr/shared/CMPT/big-data/spark-3.1.2-bin-hadoop3.2/bin/:${PATH}
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PYSPARK_PYTHON=python3
```

In datasets unzip original_tax_climate.csv.zip


Second,run this line open the file for output:

```
spark-submit clean_tax_climate.py datasets/original_tax_climate.csv cleaned_tax_climate.csv
```
The spark file is not necessary for the related notebook to work.


### Order of program execution
run the following before any notebooks as follows:

```
python3 extract_gas_diesel_prices.py
```


### Datasets
Datasets used for the project are stored in "Datasets" folder.
