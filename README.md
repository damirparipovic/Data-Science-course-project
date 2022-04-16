## Setup:

- To setup programming runs, Python 3.5+, Jupyter Notebook, and PySpark are required to install on your machine.
- You need to import python libraries before running the programs. These are numpy, pandas, matplotlib, datetime, seaborn, scipy, sklearn libraries. (See syntax in Libraries section)


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

### Running Spark jobs
The PySpark is should in Linux-like environment. First, open the terminal(command-prompt), copy and paste these lines:

```
PATH=/usr/shared/CMPT/big-data/spark-3.1.2-bin-hadoop3.2/bin/:${PATH}
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PYSPARK_PYTHON=python3
```

Second,run this line open the file for output:

```
spark-submit spark-submit clean_tax_climate.py datasets/environmental_tax_OECD.csv cleaned_datasets/cleaned_tax_climate.csv
```


### Datasets
Datasets are stored in "Datasets" folder with
