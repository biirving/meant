import numpy as np
import pandas as pd
import sys, time, os
from tqdm import tqdm

# Read CSV files
dija_price_csv = pd.read_csv('Combined_News_DJIA.csv')
price_csv = pd.read_csv('DJIA_table.csv')

price_djia = pd.merge(dija_price_csv, price_csv, on='Date', how='inner')

# ParameterAs
high_ratio = 0.0055
low_ratio = -0.005
lag_period = 5

price_djia['djia_label'] = None

negatives = 0
positives = 0 

# Loop through the prices to calculate movement ratios and classify
for i in tqdm(range(len(price_djia) - 1)):
    first_close = price_djia.iloc[i]['Adj Close']
    last_close = price_djia.iloc[i + 1]['Adj Close']
    movement_ratio = (last_close - first_close) / first_close

    if movement_ratio >= high_ratio:
        label = 1
        positives += 1
    elif movement_ratio <= low_ratio:
        label = 0
        negatives += 1
    else:
        label = None
    
    price_djia.at[i, 'djia_label'] = label

print('Positives', positives)
print('Negatives', negatives)

# Convert lists to DataFrames
columns_to_shift = [col for col in price_djia.columns if col not in ['Date', 'djia_label', 'label']]
# Create shifted columns for text and numeric columns
shifted_dfs = []
for i in range(5):
    shifted_df = price_djia[columns_to_shift].shift(i).add_suffix(f'_{4 - i}')
    shifted_dfs.append(shifted_df)

# Create shifted columns for dates
for i in range(5):
    aux_date_col = price_djia['Date'].shift(i).rename(f'aux_date_{4 - i}')
    shifted_dfs.append(aux_date_col)
# Concatenate the original date, label, text, ticker with shifted columns
result_df = pd.concat([price_djia[['Date', 'djia_label']]] + shifted_dfs, axis=1)
# Drop rows with NaN values (due to shifting)
result_df = result_df.dropna().reset_index(drop=True)
result_df.to_csv('djia_news_final.csv', index=False)

#os.makedirs(os.path.dirname(output_path), exist_ok=True)