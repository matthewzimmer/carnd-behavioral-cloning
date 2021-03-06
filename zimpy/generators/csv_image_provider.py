import random

import cv2
import numpy as np
from zimpy.camera_preprocessor import preprocess_image
from scipy import misc
import os
import math


def load_image(imagepath):
    path, file_name = os.path.split(imagepath)
    imagepath = 'IMG/' + file_name
    # image_array = cv2.imread(imagepath, 1)
    image_array = misc.imread(imagepath)
    if image_array is None:
        print('File Not Found: {}'.format(imagepath))
    return image_array


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    # print(random_bright)
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def batch_generator(X, Y, label, num_epochs, batch_size=32, output_shape=None, flip_images=True, classifier=None):
    population = len(X)
    counter = 0
    _index_in_epoch = 0
    _tot_epochs = 0
    batch_size = min(batch_size, population)
    batch_count = int(math.ceil(population / batch_size))

    print('Batch generating against the {} dataset with population {} and shape {}'.format(label, population, X.shape))
    while True:
        counter += 1
        print('batch gen iter {}'.format(counter))
        for i in range(batch_count):
            start_i = _index_in_epoch
            _index_in_epoch += batch_size
            if _index_in_epoch >= population:
                # Save the classifier to support manual early stoppage
                if classifier is not None:
                    classifier.save()
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

            for j in range(start_i, end_i):
                steering_angle = Y[j]
                image_path = None

                left = X[j].split(':')[0]
                center = X[j].split(':')[1]
                right = X[j].split(':')[2]
                z_score = X[j].split(':')[3]

                # print('angle: {}        z_score: {}'.format(steering_angle, z_score))

                mode = 1
                if mode == 1:
                    image_path = center  # center camera
                else:
                    # This algorithm was inspired by John Chen's algorithm
                    if steering_angle < -0.01:
                        chance = random.random()
                        if chance > 0.75:
                            image_path = left
                            augmented_steering = steering_angle*3.0
                            # print('L1 - real:{} - aug:{} - LEFT 3x'.format(steering_angle, augmented_steering))
                            steering_angle = augmented_steering
                        else:
                            if chance > 0.5:
                                image_path = left
                                augmented_steering = steering_angle*2.0
                                # print('L2 - real:{} - aug:{} - LEFT 2x'.format(steering_angle, augmented_steering))
                                steering_angle = augmented_steering
                            else:
                                if chance > 0.25:
                                    image_path = center
                                    augmented_steering = steering_angle*1.5
                                    # print('L3 - real:{} - aug:{} - CENTER 1.5x'.format(steering_angle, augmented_steering))
                                    steering_angle = augmented_steering
                                else:
                                    # progressively increase chances of introducing raw center
                                    if True or random.random() > (1. - _tot_epochs / num_epochs):
                                        # print('L4 - {} - CENTER'.format(steering_angle))
                                        image_path = center
                                    else:
                                        print('L5 - {} - SKIPPED'.format(steering_angle))
                    # else:
                        if steering_angle > 0.01:
                            chance = random.random()
                            if chance > 0.75:
                                image_path = right
                                augmented_steering = steering_angle*3.0
                                # print('R1 - real:{} - aug:{} - RIGHT 3x'.format(steering_angle, augmented_steering))
                                steering_angle = augmented_steering
                            else:
                                if chance > 0.5:
                                    image_path = right
                                    augmented_steering = steering_angle*2.0
                                    # print('R2 - real:{} - aug:{} - RIGHT 2x'.format(steering_angle, augmented_steering))
                                    steering_angle = augmented_steering
                                else:
                                    if chance > 0.25:
                                        image_path = center
                                        augmented_steering = steering_angle*1.5
                                        # print('R3 - real:{} - aug:{} - CENTER 1.5x'.format(steering_angle, augmented_steering))
                                        steering_angle = augmented_steering
                                    else:
                                        if True or random.random() > (1. - _tot_epochs / num_epochs):
                                            image_path = center
                                            # print('R4 - real:{} - aug:{} - CENTER 1x'.format(steering_angle, steering_angle))
                                        else:
                                            print('R5 - {} - SKIPPED'.format(steering_angle))
                        else:
                            # progressively increase chances of introducing raw center
                            if True or random.random() > (1. - _tot_epochs / num_epochs):
                                # print('C1 - {} - CENTER'.format(steering_angle))
                                image_path = center
                            else:
                                print('C2 - {} - SKIPPED'.format(steering_angle))

                if image_path is not None:
                    image = load_image(image_path)
                    if image is not None:
                        image = preprocess_image(image, output_shape=output_shape)
                        if flip_images and random.random() > 0.5:
                            X_batch.append(np.fliplr(image))
                            y_batch.append(-steering_angle)
                        else:
                            X_batch.append(image)
                            y_batch.append(steering_angle)

            yield np.array(X_batch), np.array(y_batch)
