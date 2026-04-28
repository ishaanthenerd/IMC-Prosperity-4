import pandas as pd
import numpy as np

trades = pd.read_csv('C:/Users/Ellison/Documents/GitHub/IMC-Prosperity-4/algo_exploration/merged_tr4.csv', sep=';')
prices = pd.read_csv('C:/Users/Ellison/Documents/GitHub/IMC-Prosperity-4/algo_exploration/merged_r4.csv', sep=';')

def rolling_slope(x):
    n = len(x)
    sum_x = n * (n - 1) / 2
    sum_y = np.sum(x)
    sum_xy = np.sum(np.arange(n) * x)
    sum_x2 = n * (n - 1) * (2 * n - 1) / 6
    denom = n * sum_x2 - sum_x**2
    if denom == 0: return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom

print('Calculating slopes...')
prices['slope_200'] = prices.groupby(['day', 'product'])['mid_price'].transform(lambda s: s.rolling(200).apply(rolling_slope, raw=True))

prices_clean = prices[['day', 'timestamp', 'product', 'mid_price', 'slope_200']].dropna()
merged = pd.merge(trades, prices_clean, left_on=['timestamp', 'symbol'], right_on=['timestamp', 'product'], how='inner')

for prod in ['VELVETFRUIT_EXTRACT', 'VEV_4000', 'VEV_5200', 'VEV_5300', 'HYDROGEL_PACK']:
    prod_trades = merged[merged['symbol'] == prod]
    for trader in ['Mark 01', 'Mark 14', 'Mark 55']:
        buys = prod_trades[prod_trades['buyer'] == trader]
        sells = prod_trades[prod_trades['seller'] == trader]
        if len(buys) > 0:
            print(f"[{prod}] {trader} BUYS ({len(buys)}): avg={buys.slope_200.mean():.4f}, min={buys.slope_200.min():.4f}, max={buys.slope_200.max():.4f}")
        if len(sells) > 0:
            print(f"[{prod}] {trader} SELLS ({len(sells)}): avg={sells.slope_200.mean():.4f}, min={sells.slope_200.min():.4f}, max={sells.slope_200.max():.4f}")
