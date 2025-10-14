import cv2
import torch
import mediapipe as mp
import numpy as np
from src.models.models import LSTMClassifier
import json

# Device selection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load label map
LABEL_MAP_PATH = '../data/metadata/labels.json'  # Update path as needed
with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)
index_to_sign = {v: k for k, v in label_map.items()}

# Model parameters (update as needed)
SEQ_LEN = 384  # used 384 for training
INPUT_DIM = 3 * 543  # 543 landmarks, each with x, y, z
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_CLASSES = len(label_map)
MODEL_PATH = '../models/asl_checkpoint_1.pth'
  
# Load model
model = LSTMClassifier(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

mp_holistic = mp.solutions.holistic # type: ignore

# Buffer for sequence frames
SEQUENCE_LENGTH = SEQ_LEN
sequence_buffer = []


def extract_landmarks(results):
    # Extract holistic landmarks (pose, face, left/right hand)
    landmarks = []
    for lm_type in [results.pose_landmarks, results.face_landmarks,
                    results.left_hand_landmarks, results.right_hand_landmarks]:
        if lm_type:
            for lm in lm_type.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            # If missing, pad with zeros
            if lm_type == results.face_landmarks:
                landmarks.extend([0.0, 0.0, 0.0] * 468)
            elif lm_type == results.pose_landmarks:
                landmarks.extend([0.0, 0.0, 0.0] * 33)
            elif lm_type == results.left_hand_landmarks or lm_type == results.right_hand_landmarks:
                landmarks.extend([0.0, 0.0, 0.0] * 21)
    return landmarks


def run_realtime():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            landmarks = extract_landmarks(results)
            sequence_buffer.append(landmarks)
            if len(sequence_buffer) > SEQUENCE_LENGTH:
                sequence_buffer.pop(0)
            # Only predict if buffer is full
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                input_seq = np.array(sequence_buffer, dtype=np.float32)
                input_tensor = torch.tensor(input_seq).unsqueeze(0).to(DEVICE)  # (1, seq_len, input_dim)
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_idx = torch.argmax(output, dim=1).item()
                    sign = index_to_sign.get(pred_idx, 'Unknown')
                cv2.putText(frame, f'Prediction: {sign}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Sign Language Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_realtime()
