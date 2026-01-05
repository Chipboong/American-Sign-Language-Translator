import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ============================
# 1. Load Dataset from CSV
# ============================

def load_dataset_from_csv(csv_path):
    dataset_dir = os.path.dirname(csv_path)
    df = pd.read_csv(csv_path)

    # Build class mapping
    signs = sorted(df["sign"].unique())
    class_to_index = {s: i for i, s in enumerate(signs)}

    # Full paths
    file_paths = [os.path.join(dataset_dir, p).replace("\\","/") for p in df["path"]]
    labels = [class_to_index[s] for s in df["sign"]]

    return file_paths, labels, class_to_index

# ============================
# 2. Load Parquet Keypoints
# ============================

def load_parquet_keypoints(parquet_path, target_frames=60):
    """
    Load keypoints from parquet and produce (target_frames, 543, 2).
    Handles NaN (set to 0) and pads sequences shorter than target_frames with zeros.
    """
    df = pd.read_parquet(parquet_path)

    # Replace NaN with 0
    df["x"] = df["x"].fillna(0)
    df["y"] = df["y"].fillna(0)

    # Type offsets
    type_order = ["face", "left_hand", "pose", "right_hand"]
    type_counts = {"face": 468, "left_hand": 21, "pose": 33, "right_hand": 21}
    offsets = {}
    off = 0
    for t in type_order:
        offsets[t] = off
        off += type_counts[t]

    # Frame IDs
    frame_ids = sorted(df["frame"].unique())
    all_frames = []

    for f in frame_ids:
        rows = df[df["frame"] == f]

        kpts = np.zeros((543, 2), dtype=np.float32)  # initialize with zeros

        for _, r in rows.iterrows():
            t_raw = r["type"]
            if t_raw not in offsets:
                continue
            idx_global = offsets[t_raw] + int(r["landmark_index"])
            if idx_global < 543:
                kpts[idx_global, 0] = r["x"]
                kpts[idx_global, 1] = r["y"]

        all_frames.append(kpts)

    # Pad with zeros if fewer frames than target_frames
    num_to_pad = target_frames - len(all_frames)
    if num_to_pad > 0:
        padding = [np.zeros((543, 2), dtype=np.float32) for _ in range(num_to_pad)]
        all_frames.extend(padding)

    # Sample frames if more than target_frames
    elif len(all_frames) > target_frames:
        idx = np.linspace(0, len(all_frames)-1, target_frames, dtype=int)
        all_frames = [all_frames[i] for i in idx]

    return np.array(all_frames, dtype=np.float32)  # (target_frames, 543, 2)

def select_27_keypoints(kpts_543):
    selected_indices = [489, 491, 494, 500, 501, 502, 503,
                        468, 472, 473, 476, 477, 480, 481,
                        484, 485, 488, 522, 526, 527, 530,
                        531, 534, 535, 538, 539, 542]  # 27
    return kpts_543[:, selected_indices, :]

# ============================
# 4. Normalize Keypoints
# ============================

def normalize_keypoints(keypoints):
    """
    Shoulder-based normalization.
    """
    shoulder_l_idx = 3
    shoulder_r_idx = 4

    shoulder_l = keypoints[:, shoulder_l_idx, :]
    shoulder_r = keypoints[:, shoulder_r_idx, :]
    
    # Center (midpoint of shoulders)
    center = (shoulder_l + shoulder_r) / 2
    center = center.mean(axis=0)
    
    # Scale by shoulder distance
    mean_dist = np.mean(np.linalg.norm(shoulder_l - shoulder_r, axis=1))
    if mean_dist > 0:
        keypoints = keypoints - center
        keypoints = keypoints / mean_dist
    
    return keypoints.astype(np.float32)

# ============================
# 5. Preprocess Dataset
# ============================

def preprocess_dataset(csv_path, output_dir="processed_stgcn", target_frames=60, test_size=0.1, val_size=0.1):
    file_paths, labels, class_to_index = load_dataset_from_csv(csv_path)

    # 80-10-10 split
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        file_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, test_size=val_size/(1-test_size), random_state=42, stratify=train_val_labels
    )

    dataset_splits = {
        "train": (train_files, train_labels),
        "val": (val_files, val_labels),
        "test": (test_files, test_labels)
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save class mapping
    with open(output_dir / "class_index.json", "w") as f:
        json.dump(class_to_index, f, indent=4)

    metadata = {
        "num_classes": len(class_to_index),
        "num_nodes": 27,
        "num_channels": 2,
        "num_frames": target_frames,
        "format": "(num_frames, num_nodes, num_channels)",
        "class_to_idx": class_to_index,
        "idx_to_class": {v:k for k,v in class_to_index.items()},
        "classes": [k for k,v in class_to_index.items()]
    }

    for split, (files, lbls) in dataset_splits.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        processed_labels = []
        failed_files = []

        print(f"\nProcessing {split} ({len(files)} samples)...")
        for i, (fp, lbl) in enumerate(tqdm(zip(files, lbls), total=len(files))):
            try:
                kpts_543 = load_parquet_keypoints(fp, target_frames)
                kpts_27 = select_27_keypoints(kpts_543)
                kpts_norm = normalize_keypoints(kpts_27)

                np.save(split_dir / f"stgcn_{i:04d}.npy", kpts_norm)
                processed_labels.append(lbl)
            except Exception as e:
                failed_files.append(fp)
                print(f"Failed: {fp} -> {e}")

        # Save labels
        np.save(output_dir / f"{split}_labels.npy", np.array(processed_labels))

        # Save failed files
        with open(split_dir / f"{split}_failed.txt", "w") as f:
            for ff in failed_files:
                f.write(ff + "\n")

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Preprocessing complete!")
    print(f"Metadata saved at {output_dir / 'metadata.json'}")
    return metadata

# ============================
# 6. Example usage
# ============================

if __name__ == "__main__":
    csv_path = "../dataset/train_filterd_subset_20.csv"
    output_dir = "../processed_stgcn_4"
    metadata = preprocess_dataset(csv_path, output_dir=output_dir, target_frames=60)
