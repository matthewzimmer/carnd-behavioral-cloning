import os
import csv
import shutil
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.applications import VGG16
from keras.layers import Dense, Flatten, Dropout, ELU
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import math
import cv2
import pickle
from scipy import misc, random
from sklearn.model_selection import train_test_split

from training import TrainTrackA, CommaAI, SimpleConvnet, Nvidia, Udacity, Basic
from zimpy.camera_preprocessor import preprocess_image, predict_images
from zimpy.generators.csv_image_provider import batch_generator, load_image
from zimpy.serializers.trained_data_serializer import TrainedDataSerializer

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('network_arch', 'commaai', "The network architecture to train on.")
flags.DEFINE_integer('epochs', 1, "The number of epochs.")
flags.DEFINE_integer('batch_size', 32, "The batch size.")
flags.DEFINE_integer('samples_per_epoch', None, "The number of samples per epoch during training.")
flags.DEFINE_integer('algo_mode', 5, "The algorithm to train against.")
flags.DEFINE_boolean('repickle', True, "Whether to regenerage the train.p file of training camera images.")
flags.DEFINE_boolean('use_weights', False, "Whether to use prior trained weights.")
flags.DEFINE_float('lr', 0.0001, "Optimizer learning rate.")

train_samples_seen = []
X_train, y_train, X_val, y_val = None, None, None, None
img_rows, img_cols = None, None


def move_training_images(classifier):
    drive_log_path = './driving_log.csv'
    img_path = './IMG'
    shutil.move(drive_log_path, drive_log_path + '_' + classifier.uuid)
    shutil.move(img_path, img_path + '_' + classifier.uuid)
    # os.remove(drive_log_path)


def load_track_csv():
    X_train, y_train = [], []

    # Only look at latest driving_log.csv
    drive_log_path = './driving_log.csv'

    if os.path.isfile(drive_log_path):
        df = pd.read_csv(drive_log_path)
        headers = list(df.columns.values)
        print(headers)
        for index, row in df.iterrows():
            c = row['center'].strip()
            l = row['left'].strip()
            r = row['right'].strip()
            a = float(row['steering'])

            if os.path.isfile(c):
                # casts absolute path to relative to remain env agnostic
                l, c, r = [('IMG/' + os.path.split(file_path)[1]) for file_path in (l, c, r)]
                # single string in memory
                x = '{}:{}:{}'.format(l, c, r)
                X_train.append(x)
                y_train.append(a)

    # Split some of the training data into a validation dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.01,
        random_state=0)

    X_train, y_train, X_val, y_val = np.array(X_train), np.array(y_train, dtype=np.float32), np.array(X_val), np.array(
        y_val, dtype=np.float32)

    return X_train, y_train, X_val, y_val


def main(_):
    output_shape = (40, 80, 3)
    X_train, y_train, X_val, y_val = load_track_csv()

    print('population: ', len(X_train))

    # train model
    clf = Basic()
    model = clf.get_model(input_shape=output_shape, output_shape=output_shape, use_weights=FLAGS.use_weights,
                          learning_rate=FLAGS.lr)

    samples_per_epoch = len(X_train)
    if FLAGS.samples_per_epoch is not None:
        print('overriding samples per epoch from {} to {}'.format(samples_per_epoch, FLAGS.samples_per_epoch))
        samples_per_epoch = FLAGS.samples_per_epoch

    # A data generator was used to generate more data for better accuracy
    train_datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.02,
        fill_mode='nearest')

    # history = model.fit_generator(train_generator,
    history = model.fit_generator(
        batch_generator(X=X_train, Y=y_train, label='train set', num_epochs=FLAGS.epochs, flip_images=True,
                        batch_size=FLAGS.batch_size,
                        output_shape=output_shape),
        nb_epoch=FLAGS.epochs,
        samples_per_epoch=samples_per_epoch,
        validation_data=None,
        verbose=2)

    print(history.history)
    clf.save()

    # move_training_images(clf)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
