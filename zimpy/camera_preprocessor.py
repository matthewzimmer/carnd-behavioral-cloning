import cv2
import numpy as np


def flip_image(image_array, steering_angle):
    return np.fliplr(image_array), -steering_angle


def preprocess_image(image_array, output_shape=(160, 320)):
    # hard-code this so drive.py and training_test.py use same size (refactor later)
    output_shape = (16, 32)
    # output_shape = (32, 64)
    # output_shape = (160, 320)

    # 1. Resize/normalize to desired shape
    image_array = cv2.resize(image_array, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_AREA) / 255.0
    # image_array = cv2.resize(image_array, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_AREA) / 127.5 - 1
    # image_array = cv2.normalize(image_array, image_array, norm_type=cv2.NORM_MINMAX)

    # 2. crop top third of image
    # h, w = image_array.shape[0:2]
    # y1 = int(h/3)
    # y2 = h
    # image_array = image_array[y1:y2, 0:w]

    return image_array
