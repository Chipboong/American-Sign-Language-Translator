import os
import shutil
import pandas as pd

# Paths
CSV_PATH = 'train.csv'
DST_ROOT = 'train/after'

# Read CSV
df = pd.read_csv(CSV_PATH)

# Filter for 'after' sign
after_df = df[df['sign'] == 'after']

# Copy files
for _, row in after_df.iterrows():
    src_path = row['path']  # Use path directly from CSV
    participant_id = str(row['participant_id'])
    sequence_id = str(row['sequence_id'])
    dst_dir = os.path.join(DST_ROOT, participant_id)
    dst_path = os.path.join(dst_dir, f"{sequence_id}.parquet")
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src_path, dst_path)

# List all participant_id folders and sequence_ids
print("Participants in 'after' folder:")
for participant_id in sorted(os.listdir(DST_ROOT)):
    participant_dir = os.path.join(DST_ROOT, participant_id)
    if os.path.isdir(participant_dir):
        sequence_files = sorted(os.listdir(participant_dir))
        sequence_ids = [f.replace('.parquet', '') for f in sequence_files]
        print(f"  {participant_id}: {sequence_ids}")