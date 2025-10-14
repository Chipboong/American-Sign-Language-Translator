# Isolated Sign Language Recognition (Google ASL Competition)

This project is a deep learning pipeline for isolated sign language recognition using hand, face, and pose landmarks. It is inspired by the Google ASL competition and supports training, evaluation, and real-time inference using webcam input.

## Features
- Modular PyTorch codebase
- LSTM-based sequence classifier (easy to extend to CNN/Transformer)
- Data loader for MediaPipe landmark `.parquet` files
- Training, evaluation, and real-time webcam inference
- Progress bars and clean console UI
- Dataset folder is ignored by git

## Usage
1. **Prepare the dataset**
   - Place your landmark `.parquet` files in the `dataset/train_landmark_files/` directory.
   - Place your metadata CSVs (e.g., `train.csv`, `val.csv`) in `src/data/metadata/`.
   - Place your label map JSON (e.g., `labels.json`) in `src/data/metadata/`.

2. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```
   (You may need to manually install `torch`, `tqdm`, `tabulate`, `mediapipe`, etc.)

3. **Run the project**
   ```bash
   python main.py
   ```
   - Choose Train, Evaluate, or Real-time Inference from the menu.

## Dataset Folder Structure
```
workspace/
├── dataset/
|   └── labels.json
│   └── train_landmark_files/
│       ├── <participant_id>/
│       │   ├── <sequence_id>.parquet
│       │   └── ...
│       └── ...
├── src/
│   └── data/
│       └── metadata/
│           ├── train.csv
│           |── val.csv
│           
```
- Each `.parquet` file contains landmark data for one sign sequence.
- `train.csv` and `val.csv` contain metadata with columns: `path`, `participant_id`, `sequence_id`, `sign`.
- `labels.json` maps sign names to class indices.

## How to Train
- Select "Train" in the menu to start training.
- Model checkpoints are saved as `asl_checkpoint_1.pth`.

## How to Evaluate
- Select "Evaluate" in the menu to compute accuracy on the validation set.

## How to Run Real-time Inference
- Select "Real-time Inference" in the menu to use your webcam for live sign prediction.

## License
This project is for educational and research purposes only.
