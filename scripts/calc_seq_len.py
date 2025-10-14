import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import os
from tqdm import tqdm

CSV_PATH = '../src/data/metadata/train.csv'  # Update path as needed
DATA_ROOT = '../dataset'  # Update path as needed

# Read CSV
df = pd.read_csv(CSV_PATH)
lengths = []
max_len = 0
max_path = None

for path in tqdm(df['path'], desc='Processing samples'):
    parquet_path = os.path.join(DATA_ROOT, path)
    try:
        data = pd.read_parquet(parquet_path)
        seq_len = len(data)
        lengths.append(seq_len)
        if seq_len > max_len:
            max_len = seq_len
            max_path = parquet_path
    except Exception as e:
        print(f"Error reading {parquet_path}: {e}")

if lengths:
    median_len = int(np.median(lengths))
    mean_len = int(np.mean(lengths))
    p90_len = int(np.percentile(lengths, 90))
    print(f"Median sequence length: {median_len}")
    print(f"Mean sequence length: {mean_len}")
    print(f"90th percentile sequence length: {p90_len}")
    print(f"Max sequence length: {max_len}")
    print(f"Sample with max length: {max_path}")
else:
    print("No valid samples found.")
