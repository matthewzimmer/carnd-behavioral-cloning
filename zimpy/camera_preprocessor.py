import cv2
import numpy as np
from scipy import misc

def flip_image(image_array, steering_angle):
    return np.fliplr(image_array), -steering_angle


def preprocess_image(image_array, output_shape=(160, 320)):
    # hard-code this so drive.py and training_test.py use same size (refactor later)
    # output_shape = (66, 200)
    # output_shape = (16, 32)
    # output_shape = (32, 64)
    output_shape = (160, 320)

    # 1. Resize/normalize to desired shape
    image_array = cv2.resize(image_array, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_AREA)
    # image_array = cv2.resize(image_array, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_AREA) / 255.0
    # image_array = cv2.resize(image_array, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_AREA) / 127.5 - 1
    # image_array = cv2.normalize(image_array, image_array, norm_type=cv2.NORM_MINMAX)

    # 2. crop top third of image
    h, w = image_array.shape[0:2]
    y1 = int(h/2)
    y2 = h
    image_array = image_array[y1:y2, 0:w]

    return image_array


def predict_images(model):
    images = [
        # ('/Users/matthewz/git/udacity/carnd/carnd-behavioral-cloning/IMG/center_2016_12_12_14_25_04_974.jpg', -0.1769547),
        # ('/Users/matthewz/git/udacity/carnd/carnd-behavioral-cloning/IMG/center_2016_12_12_14_25_00_642.jpg', 0.1575889),
        # ('/Users/matthewz/git/udacity/carnd/carnd-behavioral-cloning/IMG/center_2016_12_12_14_48_33_665.jpg', 0),
        # ('/Users/matthewz/git/udacity/carnd/carnd-behavioral-cloning/IMG/center_2016_12_12_14_48_34_811.jpg', -0.01234567),
        # ('/Users/matthewz/git/udacity/carnd/carnd-behavioral-cloning/IMG/center_2016_12_12_14_48_38_968.jpg', -0.1479061),
    ]

    for image_tup in images:
        image_array = misc.imread(image_tup[0])
        image_array = preprocess_image(image_array)
        pred = float(model.predict(image_array[None, :, :, :], batch_size=1))
        true = float(image_tup[1])
        print('P: {}    T: {}'.format(pred, true))