from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('asl_cnn_model.h5')

# Load and preprocess the images
img_paths = ['my_handsign2.jpg']  # List of image paths
img_arrays = []
for image_path in img_paths:
  # Load and preprocess each image
  # Image input size should match the model's expected input size
  img = image.load_img(image_path, target_size=(224, 224))
  img_array = image.img_to_array(img)
  img_array = img_array / 255.0  # Normalize if pixel values instead of 0-255, use [0, 1]
  img_arrays.append(img_array)

batch_array = np.array(img_arrays)  # Shape: (N, 224, 224, 3)

# Predict
predictions = model.predict(batch_array) # Get prediction probabilities
predicted_class = np.argmax(predictions, axis=1) # Get the index of the highest probability

# class_indices dictionary
class_indices = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'Nothing': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'Space': 20, 'T': 21, 'U': 22, 'V': 23, 'W': 24, 'X': 25, 'Y': 26, 'Z': 27}
# Reverse the dictionary
index_to_class = {v: k for k, v in class_indices.items()}

predicted_classes = [index_to_class[idx] for idx in predicted_class]
print("Predicted class:", predicted_classes)