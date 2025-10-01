import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import os

# Path to dataset
train_dir = "ASL_Dataset/Train"
val_dir = "ASL_Dataset/Validation"   # Create validation split

# Image generator with augmentation (same size for train and val)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2   # 20% for validation
)

# Train and validation generators
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)
val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)
# Number of classes
num_classes = len(train_gen.class_indices)
print("Classes:", train_gen.class_indices)

# Load pretrained MobileNetV2 (feature extractor)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False   # freeze base layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

# Compile the Model (configuring how the model will learn)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train the Model (teach the CNN to recognize ASL signs)
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15
)

# Evaluate the Model (check how well the model learned)
loss, acc = model.evaluate(val_gen)
print(f"✅ Validation Accuracy: {acc:.2f}")

model.save("asl_cnn_model.keras")

