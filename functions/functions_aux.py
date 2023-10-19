#-------------------------------------------------------
# @author Helber
#-------------------------------------------------------

# Importing libraries needed for Operating System Manipulation in Python
import platform, psutil

# Importing library for manipulation and exploration of datasets.
import numpy as np
import pandas as pd
#-------------------------------------------------------

# Function for System Information

# Credits: https://thepythoncode.com/article/get-hardware-system-information-python
def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def info_system():
    print("=" * 40, "System Information", "=" * 40)
    uname = platform.uname()
    print(f"System: {uname.system}")
    print(f"Release: {uname.release}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")

    # let's print CPU information
    print("=" * 40, "CPU Info", "=" * 40)
    # number of cores
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Total cores:", psutil.cpu_count(logical=True))
    # CPU frequencies
    cpufreq = psutil.cpu_freq()

    print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    print(f"Min Frequency: {cpufreq.min:.2f}Mhz")

    # Memory Information
    print("=" * 40, "Memory Information", "=" * 40)

    # get the memory details
    svmem = psutil.virtual_memory()
    print(f"Total: {get_size(svmem.total)}")
#-------------------------------------------------------
# Functions to calculate differences in datasets

def summary_quick_in_between_datasets(x, y):
    print(f'First dataset, get the number of rows and columns: {x.shape}')
    print(f'Get the number of elements: {x.size}', end=' ')
    print('\n')
    print(f'First dataset, get the number of rows and columns: {y.shape}')
    print(f'Get the number of elements: {y.size}')

    print('-' * 70)

    rows, cols = abs(x.shape[0] - y.shape[0]), abs(abs(x.shape[1] - y.shape[1]))
    print(f'Difference between datasets in rows: {rows} and columns: {cols}')

# Creating a function to calculate the differences in train and test data
def diff(x, y):
    # Creating an empty list, to store the results
    result = []

    # loop to traverse dataframe from train
    for i in x:
        # loop to loop through test dataframe
        for j in y:
            # Calculates the difference between the values with abs function calls
            calcule = abs(i - j)

        # add the results obtained in the list
        result.append(calcule)

    #Returns the listwith difference results
    return result
# -------------------------------------------------------

# Function to count null and missing values

def missing_values(df):

    """For each column with missing values and  the missing proportion."""
    data = [(col, df[col].isna().sum() / len(df) * 100)
            for col in df.columns if df[col].isnull().sum() > 0]
    col_names = ['column', 'percent_missing', ]

    # Create dataframe with values missing in an ordered way
    missing_df = pd.DataFrame(data, columns=col_names).sort_values('percent_missing')

    # Return dataframe the values missing
    return missing_df

def check_missing_values(X):

    # Check exists between NAN values of each variable
    nan = X.isna().sum()
    print("=" * 40, "NAN values count", "=" * 40)
    print(nan[nan > 0])

    # Creating a variable check total NAN values
    total = X.isnull().sum().sum()

    # Creating a variable check total in percentage terms
    total_perc = (X.isnull().sum() / X.shape[0]) * 100

    print("=" * 40, "Total NAN values", "=" * 40)
    print(f'\nTotal:  {total}')
    print(f'\nTotal in terms of percentages: {round(total_perc[total_perc > 0], 2).sum()} %')

def unique_nan(x):
    return x.nunique(dropna=False)

def count_nulls(x):
    return x.size - x.count()
#-------------------------------------------------------

