import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from predict_stgcn import load_trained_model
from stgcn_training import STGCNDataGenerator

# -------------------------------------------------------
# 1. LOAD MODEL
# -------------------------------------------------------
DATA_DIR = "../preprocessed_stgcn_3"
BATCH_SIZE = 32
DROPOUT = 0.2
MODEL_PATH = "../models_stgcn2/stgcn_20251127_225130/final_model_weights.weights.h5"    # <-- change this to your model path
model = load_trained_model(MODEL_PATH, 20, DROPOUT)
print("\nLoaded model:", MODEL_PATH)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

test_labels_path = os.path.join(DATA_DIR, 'test_labels.npy')
# if os.path.exists(test_labels_path):
#     test_gen = STGCNDataGenerator(
#         DATA_DIR,
#         split='test',
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         augmentation=False
#        )
# -------------------------------------------------------
# 2. LOAD TEST DATA
# -------------------------------------------------------
# X_test shape should be: (num_samples, frames, keypoints, channels)
# y_test should be integer labels (0..num_classes)
test_gen = STGCNDataGenerator(
        DATA_DIR,
        split='test',
        batch_size=BATCH_SIZE,
        shuffle=False,
        augmentation=False
       )    # <-- change path
X_test, y_test = test_gen.__getitem__(32)

print("Test data shape:", X_test.shape)

# -------------------------------------------------------
# 3. RUN EVALUATION
# -------------------------------------------------------
loss, acc = model.evaluate(test_gen, verbose=1)
print("\n========== MODEL PERFORMANCE ==========")
print(f"Test Accuracy : {acc:.4f}")
print(f"Test Loss     : {loss:.4f}")

# -------------------------------------------------------
# 4. PREDICT ON TEST SET
# -------------------------------------------------------
y_prob = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

# Get true labels
y_true = []
for _, y_batch in test_gen:
    y_true.extend(y_batch)
y_true = np.array(y_true)
if len(y_true.shape) > 1:
    y_true = np.argmax(y_true, axis=1)

# Classification report
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred))

# Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()