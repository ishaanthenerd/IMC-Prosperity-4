import pandas as pd
import numpy as np
import string, json, math, statistics
import matplotlib.pyplot as plt
from io import StringIO

'''
DATA HANDLING
'''

# Load data from all three files; add info about best/worst bid/ask
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

    bids = ["bid_price_1", "bid_price_2", "bid_price_3"]
    asks = ["ask_price_1", "ask_price_2", "ask_price_3"]
    full_df["best_bid"] = full_df[bids].max(axis=1)
    full_df["best_ask"] = full_df[asks].min(axis=1)
    full_df["worst_bid"] = full_df[bids].min(axis=1)
    full_df["worst_ask"] = full_df[asks].max(axis=1)

    return full_df

# Plot two columns of a DataFrame against each other
def plot_df(
        df: pd.DataFrame,
        column1: str,
        column2: str,
        title: str = "",
    ):

    plt.figure(figsize=(12, 6))
    plt.plot(df[column1], df[column2], linewidth=1, label=f"{column2}")
    plt.title(title)
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()

# Plot a set of columns of a DataFrame against timestamp
def plot_against_timestamp(
        df: pd.DataFrame,
        columns: list[str],
        title: str = "",
    ):

    plt.figure(figsize=(12, 6))
    for column in columns:
        plt.plot(df["timestamp"], df[column], linewidth=1, label=f"{column}")
    plt.title(title)
    plt.xlabel("timestamp")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()

# Plot mid price + other prices against timestamp
def plot_mid_price(
        df: pd.DataFrame,
        incl_best_bid: bool = False,
        incl_best_ask: bool = False,
        incl_worst_bid: bool = False,
        incl_worst_ask: bool = False,   
        title: str = "", 
    ):

    columns = ["mid_price"]
    if incl_best_bid:
        columns.append("best_bid")
    if incl_best_ask:
        columns.append("best_ask")
    if incl_worst_bid:
        columns.append("worst_bid")
    if incl_worst_ask:
        columns.append("worst_ask")
    plot_against_timestamp(df, columns, title)

def plot_orderbook_buy_size(
        df: pd.DataFrame,
        title: str = "",    
    ):

    volume_cols_bid = ["bid_volume_1", "bid_volume_2", "bid_volume_3"]
    df[volume_cols_bid] = df[volume_cols_bid].fillna(0)
    df["bid_book_size"] = df[volume_cols_bid].sum(axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["bid_book_size"], label="Bid Size")
    plt.xlabel("Timestamp")
    plt.ylabel("Orderbook Size")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_orderbook_sell_size(
        df: pd.DataFrame,
        title: str = "",    
    ):

    volume_cols_ask = ["ask_volume_1", "ask_volume_2", "ask_volume_3"]
    df[volume_cols_ask] = df[volume_cols_ask].fillna(0)
    df["ask_book_size"] = df[volume_cols_ask].sum(axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["ask_book_size"], label="Ask Size")
    plt.xlabel("Timestamp")
    plt.ylabel("Orderbook Size")
    plt.title(title)
    plt.legend()
    plt.show()


def extract_log_data(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as f:
        log_data = json.loads(f.read().replace('\n', '\\n'))
    csv_data = log_data['activitiesLog']
    return pd.read_csv(StringIO(csv_data), sep=';')

def plot_pnl(df: pd.DataFrame, title: str = "Profit and Loss"):
    plt.figure(figsize=(12, 6))

    # Plot total PnL
    total_pnl = df.groupby('timestamp')['profit_and_loss'].sum().reset_index()
    total_adj_r = calculate_adjusted_r_squared(total_pnl["timestamp"], total_pnl["profit_and_loss"])
    print(f"Adjusted R^2 for Total PnL: {total_adj_r:.4f}")
    plt.plot(total_pnl["timestamp"], total_pnl["profit_and_loss"], 
             label=f"Total PnL (adj R^2: {total_adj_r:.4f})", linewidth=2, color='black')

    # Plot PnL for each individual product
    products = df['product'].unique()
    for product in products:
        product_df = df[df['product'] == product]
        if not product_df.empty:
            adj_r = calculate_adjusted_r_squared(product_df["timestamp"], product_df["profit_and_loss"])
            print(f"Adjusted R^2 for {product}: {adj_r:.4f}")
            plt.plot(product_df["timestamp"], product_df["profit_and_loss"], 
                     label=f"{product} (adj R^2: {adj_r:.4f})", linewidth=1, alpha=0.7)
    # ... grid, legend, show
    plt.title(title)
    plt.legend()
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

# Calculate adjusted R^2
def calculate_adjusted_r_squared(x, y):
    n = len(x)
    if n <= 2: return 0.0
    coeffs = np.polyfit(x, y, 1)
    p = np.poly1d(coeffs)
    y_hat = p(x)
    y_bar = np.mean(y)
    ss_tot = np.sum((y - y_bar)**2)
    ss_res = np.sum((y - y_hat)**2)
    if ss_tot == 0: return 0.0
    r_squared = 1 - (ss_res / ss_tot)
    return 1 - (1 - r_squared) * (n - 1) / (n - 2)