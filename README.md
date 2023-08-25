# Oil Prices and Taxes Project
This project focuses on collecting data regarding oil prices and tax rates from different countries. Our primary aim is to search for potential correlations and extract meaningful insights from the data.

## Prerequisites

### System Requirements:
- **Python**: Version 3.5 or newer.
- **Jupyter Notebook**
- **PySpark**

### Python Libraries:
Make sure to install the following Python libraries before running the program:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

**Note**: While there are various environments to run `.ipynb` files, such as Anaconda or Visual Studio, this guide assumes you're using Jupyter Notebook.

## Libraries Syntax

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from scipy import stats
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
```

## Execution Guidelines

### Running Jupyter Notebooks:
- Files with `.ipynb` extension can be directly executed within the Jupyter Notebook environment.

### Running Python Scripts:
- For files with a `.py` extension, execute them via a terminal.

### Running Spark Jobs:

If you're operating in a Linux-like environment, follow these steps:

1. Configure the environment variables:
```bash
   export PATH=/usr/shared/CMPT/big-data/spark-3.1.2-bin-hadoop3.2/bin/:${PATH}
   export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
   export PYSPARK_PYTHON=python3
```

2. Unzip the required dataset:
```bash
   unzip Datasets/original_tax_climate.csv.zip
```

3. Execute the Spark job:
```bash
   spark-submit clean_tax_climate.py Datasets/original_tax_climate.csv cleaned_tax_climate.csv
```

**Note**: Executing the Spark job isn't mandatory for the related notebook to function.

### Order of Execution:

Prior to running any notebooks, execute the following:
```bash
python3 extract_gas_diesel_prices.py
```

## Datasets

The datasets used for this project are located within the `Datasets` folder.
