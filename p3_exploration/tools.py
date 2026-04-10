import pandas as pd
import numpy as np
import string, json, math, statistics
import matplotlib.pyplot as plt

'''
DATA HANDLING
'''

# Load data from all three files
def extract_data(
        cur_round: int, 
        rows_per_file: int = 80000
    ) -> pd.DataFrame:

    full_df = pd.DataFrame(columns=[])
    for day in range(cur_round - 3, cur_round):
        file_path = "r" + str(cur_round) + "_d" + str(day) + ".csv"
        df = pd.read_csv(file_path, sep=";")
        df.index = df.index + rows_per_file * (day - (cur_round - 3))
        df["timestamp"] = df["timestamp"] + 1000000 * day
        full_df = pd.concat([full_df, df])
    return full_df

# Plot two columns of a DataFrame against each other
def plot_df(
        df: pd.DataFrame,
        column1: str,
        column2: str,
        title: str = "",
    ):

    plt.figure(figsize=(12, 6))
    plt.plot(df[column1], df[column2], linewidth=1)
    plt.title(title)
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

'''
COMPUTATION
'''

# Calculate auto_correlation between columns
# If using timestamps, set column1 as "timestamp"
def auto_correlation(
        df: pd.DataFrame, 
        column1: str, 
        column2: str, 
        lag: int = 1
    ) -> float:

    if column1 not in df.columns:
        raise KeyError(f"df must contain a '{column1}' column")
    if column2 not in df.columns:
        raise KeyError(f"df must contain a '{column2}' column")

    tmp = df[[column1, column2]].copy().sort_values(column1)
    x = tmp[column2]
    y = tmp[column2].shift(lag)
    return x.corr(y)

