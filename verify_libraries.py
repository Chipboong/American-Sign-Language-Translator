# import tensorflow as tf
# #import cv2
# #import numpy as np
# #import ctypes
# print("TensorFlow:", tf.__version__)
# #print("OpenCV:", cv2.__version__)
# print(tf.test.is_gpu_available())
# print(tf.config.list_physical_devices('GPU'))
# # from tensorflow.python.client import device_lib
# # print(device_lib.list_local_devices())
import tensorflow as tf

print("TensorFlow:", tf.__version__)

# Physical devices
gpus = tf.config.list_physical_devices('GPU')
print("Physical GPUs:", gpus)

# Logical devices (after TF initializes runtimes)
print("Logical GPUs:", tf.config.list_logical_devices('GPU'))

# Optional: get device details (name, compute capability) if available
if gpus:
    try:
        details = tf.config.experimental.get_device_details(gpus[0])
        print("GPU details:", details)
    except Exception as e:
        print("Could not get GPU details:", e)
