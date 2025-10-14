import pandas as pd
from collections import defaultdict
import os

# Read train.csv
train_csv = 'train.csv'
df = pd.read_csv(train_csv)

# Group by sign and participant_id
sign_participants = defaultdict(set)
for _, row in df.iterrows():
    sign_participants[row['sign']].add(row['participant_id'])

# For each sign, select 1 participant for val set (stratified by sign)
val_participants = set()
for sign, participants in sign_participants.items():
    val_participants.add(next(iter(participants)))

# Split
val_mask = df['participant_id'].isin(val_participants)
val_df = df[val_mask]
train_df = df[~val_mask]

# Save
os.makedirs('../data/metadata', exist_ok=True)
train_df.to_csv('../data/metadata/train_split.csv', index=False)
val_df.to_csv('../data/metadata/val.csv', index=False)
print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
