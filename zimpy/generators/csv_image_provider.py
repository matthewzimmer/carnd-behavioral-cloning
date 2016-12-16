import random

import cv2
import numpy as np
from os.path import basename
from zimpy.camera_preprocessor import preprocess_image


def load_image(imagepath):
    imagepath = './IMG/'+basename(imagepath)
    img = cv2.imread(imagepath, 1)
    # do any preprocessing... resize, reshape, etc. here...
    # newx, newy = int(img.shape[1]/4),int(img.shape[0]/4) #new size (w,h)
    # newimage = cv2.resize(img,(newx,newy))
    processed_img = None
    if img is not None:
        processed_img = preprocess_image(img)
    return processed_img


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    # print(random_bright)
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def batch_generator(X, Y, label, num_epochs, batch_size=32, output_shape=None):
    population = len(X)
    counter = 0
    _index_in_epoch = 0
    start_i = 0
    _tot_epochs = 0
    print('Batch generating against the {} dataset with population {} and shape {}'.format(label, population, X.shape))
    while True:
        counter += 1
        print('batch gen iter {}'.format(counter))
        for i in range(batch_size):
            start_i = _index_in_epoch
            _index_in_epoch += batch_size
            if _index_in_epoch >= population:
                print('sampled entire population. reshuffling deck and resetting all counters.')
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
            print('  yielding train items in range {}'.format(range(start_i, end_i)))
            for j in range(start_i, end_i):
                steering_angle = Y[j]
                image_path = None
                if steering_angle < -0.01:
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
                                image_path = X[j].split(':')[1]
                else:
                    if steering_angle > 0.01:
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
                                    image_path = X[j].split(':')[1]
                    else:
                        # gradually increase our chances of intoducing
                        if random.random() > (1. - _tot_epochs / num_epochs):
                            image_path = X[j].split(':')[1]
                            steering_angle += 0.05

                if image_path is not None:
                    # print(image_path)
                    image = load_image(image_path)
                    # steering_angle = np.array([[steering_angle]])
                    # steering_angle = np.array([steering_angle])
                    # image = image.reshape(1, output_shape[0], output_shape[1], output_shape[2])
                    # image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
                    if image is not None:
                        X_batch.append(image)
                        y_batch.append(steering_angle)
                        if random.random() > 0.5:
                            # print('     >> flipping image and steering')
                            X_batch.append(np.fliplr(image))
                            y_batch.append(-steering_angle)

            yield np.array(X_batch), np.array(y_batch)
