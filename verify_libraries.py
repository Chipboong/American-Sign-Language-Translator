import tensorflow as tf
#import cv2
#import numpy as np
#import ctypes
print("TensorFlow:", tf.__version__)
#print("OpenCV:", cv2.__version__)
print(tf.test.is_gpu_available())
print(tf.config.list_physical_devices('GPU'))
