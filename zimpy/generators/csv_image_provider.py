import random

import cv2
import numpy as np
from os.path import basename
from zimpy.camera_preprocessor import preprocess_image
from scipy import misc

def load_image(imagepath):
    imagepath = 'IMG/'+basename(imagepath)
    # image_array = cv2.imread(imagepath, 1)
    # print('load image at path ', imagepath)
    image_array = misc.imread(imagepath)
    if image_array is None:
        print('File Not Found: {}'.format(imagepath))
    # print(np.array(image_array.shape))
    # print('{} shape: '.format(imagepath), image_array.shape)
    return image_array


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    # print(random_bright)
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def batch_generator(X, Y, label, num_epochs, batch_size=32, output_shape=None, flip_images=True):
    population = len(X)
    counter = 0
    _index_in_epoch = 0
    start_i = 0
    _tot_epochs = 0
    batch_size = min(batch_size, population)

    print('Batch generating against the {} dataset with population {} and shape {}'.format(label, population, X.shape))
    while True:
        counter += 1
        print('batch gen iter {}'.format(counter))
        for i in range(batch_size):
            start_i = _index_in_epoch
            _index_in_epoch += batch_size
            if _index_in_epoch >= population:
                print('  sampled entire population. reshuffling deck and resetting all counters.')
                perm = np.arange(population)
                np.random.shuffle(perm)
                X = X[perm]
                Y = Y[perm]
                start_i = 0
                _index_in_epoch = batch_size
                _tot_epochs += 1
            end_i = _index_in_epoch

            X_batch = []
            y_batch = []
            # print('  yielding train items in range {}'.format(range(start_i, end_i)))
            for j in range(start_i, end_i):
                steering_angle = Y[j]
                image_path = None

                mode = 2
                if mode == 1:
                    if steering_angle < -0.01:
                        chance = random.random()
                        if chance > 0.75:
                            image_path = X[j].split(':')[0]
                            steering_angle *= 3.0
                        else:
                            if chance > 0.5:
                                image_path = X[j].split(':')[0]
                                steering_angle *= 2.0
                            else:
                                if chance > 0.25:
                                    image_path = X[j].split(':')[1]
                                    steering_angle *= 1.5
                                else:
                                    if True or random.random() > (1. - _tot_epochs / num_epochs):
                                        image_path = X[j].split(':')[1]
                                        # steering_angle += 0.05
                    else:
                        if steering_angle > 0.01:
                            chance = random.random()
                            if chance > 0.75:
                                image_path = X[j].split(':')[2]
                                steering_angle *= 3.0
                            else:
                                if chance > 0.5:
                                    image_path = X[j].split(':')[2]
                                    steering_angle *= 2.0
                                else:
                                    if chance > 0.25:
                                        image_path = X[j].split(':')[1]
                                        steering_angle *= 1.5
                                    else:
                                        if True or random.random() > (1. - _tot_epochs / num_epochs):
                                            image_path = X[j].split(':')[1]
                                            # steering_angle += 0.05
                        else:
                            # gradually increase our chances of intoducing
                            if True or random.random() > (1. - _tot_epochs / num_epochs):
                                image_path = X[j].split(':')[1]
                                # steering_angle += 0.05
                else:
                    image_path = X[j].split(':')[1] # center camera

                if image_path is not None:
                    # print(image_path)
                    image = load_image(image_path)
                    image = preprocess_image(image, output_shape=output_shape)

                    # steering_angle = np.array([[steering_angle]])
                    # steering_angle = np.array([steering_angle])
                    # image = image.reshape(1, output_shape[0], output_shape[1], output_shape[2])
                    # image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
                    if image is not None:
                        if random.random() > 0.5:
                            # print('O')
                            X_batch.append(image)
                            y_batch.append(steering_angle)
                        else:
                            # print('  1')
                            # print('     # flipping image and steering')
                            X_batch.append(np.fliplr(image))
                            y_batch.append(-steering_angle)

            yield np.array(X_batch), np.array(y_batch)
