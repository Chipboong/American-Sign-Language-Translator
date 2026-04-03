# ASL Words Classification (ST-GCN)

American Sign Language (ASL) word classification using a **Spatial-Temporal Graph Convolutional Network (ST-GCN)** on **MediaPipe Holistic** keypoints.  
This repo includes:

- Dataset preprocessing (video → keypoints → ST-GCN tensors)
- ST-GCN training (TensorFlow/Keras)
- Offline prediction on a saved `.npy` sample
- Real-time webcam prediction with simple sign-activity detection (hand presence)

> Note: There are also older/experimental **PyTorch LSTM** scripts under `src/models/train.py` and `src/models/evaluate.py`, but the main pipeline in this repo is the **TensorFlow ST-GCN** workflow.

---

## Project structure

- `requirements.txt` — Python dependencies (TensorFlow, OpenCV, MediaPipe, etc.)
- `src/prepare_stgcn_data.py` — preprocess **raw videos** into ST-GCN `.npy` tensors
- `src/data_prepare_for_parquet.py` — preprocess from **parquet keypoints** (CSV + parquet) into `.npy`
- `src/stgcn_training.py` — train ST-GCN (TensorFlow)
- `src/predict_stgcn.py` — run inference on a single preprocessed `.npy`
- `realtime_predict.py` — webcam real-time prediction with activity detection
- `src/stgcn_augmentation.py` — augmentation (shear/rotation) for training
- `src/models/stgcn_tf.py` — ST-GCN model implementation (TensorFlow)
- `src/models/graph_utils.py` — adjacency/graph partition utilities
- `stgcn_architecture.mermaid` — ST-GCN architecture diagram

---

## Requirements

- Python **3.8+**
- A webcam (for real-time mode)
- OS: Windows / macOS / Linux should work (OpenCV backend selection is handled in `realtime_predict.py`)

Install packages:

```bash
pip install -r requirements.txt
```

---

## Data formats

This ST-GCN pipeline works on **skeleton sequences** shaped as:

- **(T, V, C)** where:
  - `T = 60` frames (this repo uses 60 in multiple scripts)
  - `V = 27` keypoints (pose + hands subset)
  - `C = 2` channels: `(x, y)`

So each sample is typically:

- `(60, 27, 2)` float32

### Keypoint extraction & normalization

From MediaPipe Holistic, the code extracts **543 landmarks** (pose + hands + face), then:

1. Selects **27 keypoints** (pose + both hands subset)
2. Applies **shoulder-based normalization**:
   - center at midpoint of shoulders
   - scale by mean shoulder distance

---

## Option A — Preprocess from raw videos (recommended if you have video dataset)

`src/prepare_stgcn_data.py` expects a dataset directory like:

```text
dataset/
  train/
    CLASS_1/
      *.mp4
    CLASS_2/
      *.mp4
  val/
    CLASS_1/
      *.mp4
  test/
    CLASS_1/
      *.mp4
```

Run preprocessing:

```bash
python src/prepare_stgcn_data.py
```

It will create an output directory (configured inside the script) like:

```text
preprocessed_stgcn/
  train/stgcn_0000.npy
  val/stgcn_0000.npy
  test/stgcn_0000.npy
  train_labels.npy
  val_labels.npy
  test_labels.npy
  metadata.json
```

---

## Option B — Preprocess from parquet keypoints + CSV

If you already have keypoints stored as parquet files and a CSV mapping them to labels, use:

- `src/data_prepare_for_parquet.py`

The CSV is expected to contain at least:
- `path` — path to parquet file (relative to the CSV folder is supported by the script)
- `sign` — class label string

Run (edit paths inside the script or copy the function into your own runner):

```bash
python src/data_prepare_for_parquet.py
```

This will output a split dataset (train/val/test) and metadata files.

---

## Training (TensorFlow ST-GCN)

Main trainer:

```bash
python src/stgcn_training.py
```

What it does:

- loads `metadata.json` + `*_labels.npy` + `train/val/test` `.npy` samples
- uses `STGCNDataGenerator` (with augmentation enabled for train split)
- trains ST-GCN + FC head
- saves:
  - `best_model_weights.weights.h5`
  - `final_model_weights.weights.h5`
  - `training_history.png`
  - `config.json` (class names, dropout, metrics, etc.)

> You may need to edit `DATA_DIR` / `OUTPUT_DIR` inside `src/stgcn_training.py` to match your local folders.

---

## Offline prediction on a single `.npy`

Use `src/predict_stgcn.py`:

```bash
python src/predict_stgcn.py \
  --data path/to/sample.npy \
  --weights path/to/best_model_weights.weights.h5 \
  --config path/to/config.json
```

Expected input shape is **(60, 27, 2)**.

The script prints:
- predicted class
- confidence
- top-k predictions
- probability table

---

## Real-time webcam prediction

Run:

```bash
python realtime_predict.py \
  --weights path/to/best_model_weights.weights.h5 \
  --config path/to/config.json
```

How it works:

- reads webcam frames
- runs MediaPipe Holistic
- **detects sign activity** using hand presence (non-zero hand keypoints)
- buffers keypoints while “signing”
- when hands disappear for `--idle_frames`, it finalizes the sign:
  - resamples sequence to `--target_frames` (default 60)
  - runs `predict_single()`
  - overlays prediction on the video feed

Controls:
- Press `q` to quit
- Press `r` to reset prediction state

Common flags:
- `--confidence_threshold` (default 0.7)
- `--idle_frames` (default 10)
- `--min_frames` (default 10)
- `--max_frames` (default 200)
- `--camera_id` (default 0)

---

## Evaluation

There is a TensorFlow evaluation script at:

- `src/evaluation.py`

It uses `STGCNDataGenerator` and prints:
- test loss / accuracy
- classification report
- confusion matrix heatmap

You will need to edit:
- `DATA_DIR`
- `MODEL_PATH`
- `DROPOUT`
- number of classes passed into `load_trained_model(...)`

Example:

```bash
python src/evaluation.py
```

---

## Notes / Tips

- If your OpenCV window does not open, ensure you installed **opencv-python** (not headless):
  ```bash
  pip install opencv-python
  ```
- Make sure `dropout` used at inference matches the training config (`config.json`).
- The ST-GCN graph is defined for **27 nodes** and is built inside:
  - `src/stgcn_training.py` (inward edges)
  - `src/predict_stgcn.py` (same edges for reconstruction)

---

## License

No license file is included in the repository yet. If you plan to share or publish this project, consider adding a LICENSE file (e.g., MIT, Apache-2.0, GPL-3.0).

---

## Citation / Acknowledgements

- MediaPipe Holistic for pose/hand/face landmarks
- ST-GCN (Spatial Temporal Graph Convolutional Networks) style architecture for skeleton-based recognition
