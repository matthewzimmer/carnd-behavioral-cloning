import cv2
import numpy as np
from scipy import misc


def flip_image(image_array, steering_angle):
    return np.fliplr(image_array), -steering_angle


def preprocess_image(image_array, output_shape=None):
    if output_shape is None:
        output_shape = (66, 200)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2YUV)

    # [y1:y2, x1:x2] - crops top portion as well as car's hood from image
    image_array = image_array[50:140, 0:320]

    # resize image to output_shape
    image_array = cv2.resize(image_array, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_AREA)

    # image_array = image_array / 255 - 0.5
    # image_array = image_array / 127.5 - 1.
    # image_array = cv2.normalize(image_array, image_array, norm_type=cv2.NORM_MINMAX)

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
