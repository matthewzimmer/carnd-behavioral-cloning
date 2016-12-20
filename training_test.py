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

from training import CommaAI, SimpleConvnet, Nvidia, Udacity, Basic, BasicELU
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
flags.DEFINE_float('dropout_prob', 0.5, "Percentage of neurons to misfire during training.")
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

    # ctr_idx = 0
    # lft_idx = 1
    # rgt_idx = 2
    # str_ang = 3

    # Only look at latest driving_log.csv
    drive_log_path = './driving_log.csv'

    if os.path.isfile(drive_log_path):
        df = pd.read_csv(drive_log_path)
        headers = list(df.columns.values)
        print(headers)
        for index, row in df.iterrows():
            # print(observation)
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


def load_track_data(output_shape, repickle=False, preprocess=True):
    pickle_file = os.path.join(os.path.dirname(__file__), 'train.p')
    if repickle == True or not os.path.isfile(pickle_file):
        X_train, y_train = [], []

        drive_log_path = './driving_log.csv'
        if os.path.isfile(drive_log_path):
            with open(drive_log_path, 'r') as drive_logs:
                has_header = csv.Sniffer().has_header(drive_logs.read(1024 * 10))
                drive_logs.seek(0)  # rewind
                incsv = csv.reader(drive_logs)
                if has_header:
                    next(incsv)  # skip header row
                observations = csv.reader(drive_logs, delimiter=',')
                for observation in observations:
                    c_image_path = observation[0].strip()
                    l_image_path = observation[1].strip()
                    r_image_path = observation[2].strip()
                    steering_angle = float(observation[3])

                    valid_images = [c_image_path]

                    # if not math.isclose(steering_angle, 0.0):
                    #     valid_images = [c_image_path]
                    # valid_images = [c_image_path, l_image_path, r_image_path]
                    # valid_images = [l_image_path, r_image_path]

                    # if steering_angle < 0:
                    #     valid_images.append(l_image_path)
                    # elif steering_angle > 0:
                    #     valid_images.append(r_image_path)

                    for image_path in valid_images:
                        if not os.path.isfile(image_path):
                            continue

                        out_image = misc.imread(image_path)
                        if preprocess == True:
                            out_image = preprocess_image(out_image, output_shape)
                        X_train.append(out_image)
                        y_train.append(steering_angle)
                        # if not math.isclose(steering_angle, 0.0) and random.random() < 0.5:
                        # if not math.isclose(steering_angle, 0.0):
                        if steering_angle > 0. or steering_angle < 0.:
                            # if True:  # random.random() < 0.5:
                            X_train.append(np.fliplr(out_image))
                            y_train.append(-1 * steering_angle)
                            # if not math.isclose(steering_angle, 0.0):
                            #     y_train.append(-1 * steering_angle)
                            # else:
                            #     y_train.append(steering_angle)

        # Split some of the training data into a validation dataset.
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.15,
            random_state=0)

        X_train, y_train, X_val, y_val = np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)
        X_train = X_train.astype('float32')
        X_val = X_val.astype('float32')

    # TrainedDataSerializer.save_data(
    #     data={'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val},
    #     pickle_file=pickle_file,
    #     overwrite=True
    # )

    # with open(pickle_file, 'wb') as pfile:
    #     pickle.dump(
    #         {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val},
    #         pfile, pickle.HIGHEST_PROTOCOL)

    else:
        data = TrainedDataSerializer.reload_data(pickle_file=pickle_file)
        X_train, y_train, X_val, y_val = data['X_train'], data['y_train'], data['X_val'], data['y_val']

    return X_train, y_train, X_val, y_val


def main(_):
    global X_train, y_train, X_val, y_val

    # if FLAGS.network_arch == 'commaai':
    #     clf = CommaAI()
    # elif FLAGS.network_arch == 'vgg16':
    #     clf = VGG16(include_top=True, weights='imagenet',
    #                   input_tensor=None, input_shape=None)
    # else:
    #     raise NotImplementedError


    input_shape = (160, 320, 3)
    output_shape = (160, 320, 3)

    # fits the model on batches with real-time data augmentation:
    train_mode = FLAGS.algo_mode
    if train_mode == 1:
        output_shape = (int(160 / 3), int(320 / 3), 3)
        X_train, y_train, X_val, y_val = load_track_data(output_shape=output_shape[0:2], repickle=FLAGS.repickle)
        # img_rows, img_cols = X_train.shape[1], X_train.shape[2]

        print('X_train shape: ', X_train.shape)
        print('X_val shape:   ', X_val.shape)

        # train model
        clf = CommaAI()
        model = clf.get_model(input_shape=output_shape, output_shape=output_shape)

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        # rotation_range=20,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # horizontal_flip=True)

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)
        samples_per_epoch = math.floor(len(X_train) / (1. * FLAGS.epochs))
        history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=FLAGS.batch_size),
                                      samples_per_epoch=len(X_train),
                                      nb_epoch=FLAGS.epochs)

    elif train_mode == 2:
        output_shape = (80, 160, 3)
        X_train, y_train, X_val, y_val = load_track_data(output_shape=output_shape[0:2], repickle=FLAGS.repickle)
        img_rows, img_cols = X_train.shape[1], X_train.shape[2]
        output_shape = (img_rows, img_cols, 3)

        print('X_train shape: ', X_train.shape)
        print('X_val shape:   ', X_val.shape)

        # train model
        clf = SimpleConvnet()
        model = clf.get_model(input_shape=output_shape, output_shape=output_shape)
        history = model.fit(X_train, y_train, batch_size=FLAGS.batch_size, nb_epoch=FLAGS.epochs,
                            validation_data=(X_val, y_val))

        # predict 3 images and compare accuracy
        predict_images(model)

    elif train_mode == 3:
        # output_shape = (16, 32, 3)
        # output_shape = (160, 320, 3)
        output_shape = (80, 160, 3)
        X_train, y_train, X_val, y_val = load_track_data(output_shape=output_shape[0:2], repickle=FLAGS.repickle)
        img_rows, img_cols = X_train.shape[1], X_train.shape[2]
        output_shape = (img_rows, img_cols, 3)

        print('X_train shape: ', X_train.shape)
        print('X_val shape:   ', X_val.shape)

        # train model
        clf = Nvidia()
        model = clf.get_model(input_shape=output_shape, output_shape=output_shape, learning_rate=FLAGS.lr)
        history = model.fit(X_train, y_train, batch_size=FLAGS.batch_size, nb_epoch=FLAGS.epochs,
                            validation_data=(X_val, y_val))

        # predict 3 images and compare accuracy
        predict_images(model)

    elif train_mode == 4:
        # train model
        clf = CommaAI()
        model = clf.get_model(input_shape=(80, 160, 3), output_shape=(80, 160, 3))

        # this is the augmentation configuration we will use for training
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        history = model.fit_generator(
            datagen.flow_from_directory('data/training/5/IMG', batch_size=32, target_size=(160, 80),
                                        class_mode='sparse'), samples_per_epoch=len(X_train), nb_epoch=FLAGS.epochs)
    elif train_mode == 5:
        output_shape = (66, 200, 3)
        #output_shape = (160, 320, 3)
        X_train, y_train, X_val, y_val = load_track_csv()

        # train model
        clf = Nvidia()
        model = clf.get_model(input_shape=output_shape, output_shape=output_shape, use_weights=FLAGS.use_weights, dropout_prob=FLAGS.dropout_prob)

        samples_per_epoch = len(X_train)
        if FLAGS.samples_per_epoch is not None:
            print('overriding samples per epoch from {} to {}'.format(samples_per_epoch, FLAGS.samples_per_epoch))
            samples_per_epoch = FLAGS.samples_per_epoch

        history = model.fit_generator(
            batch_generator(X_train, y_train, 'train set', FLAGS.epochs, batch_size=FLAGS.batch_size,
                            output_shape=output_shape),
            nb_epoch=FLAGS.epochs,
            samples_per_epoch=samples_per_epoch,
            nb_val_samples=len(X_val),
            validation_data=batch_generator(X_val, y_val, 'validation set', num_epochs=FLAGS.epochs,
                                            batch_size=FLAGS.batch_size, output_shape=output_shape),
            verbose=2)

    elif train_mode == 6:
        output_shape = (40, 80, 3)
        X_train, y_train, X_val, y_val = load_track_csv()

        print('population: ', len(X_train))

        # train model
        clf = Basic()
        model = clf.get_model(input_shape=output_shape, output_shape=output_shape, use_weights=FLAGS.use_weights, dropout_prob=FLAGS.dropout_prob)

        samples_per_epoch = len(X_train)
        if FLAGS.samples_per_epoch is not None:
            print('overriding samples per epoch from {} to {}'.format(samples_per_epoch, FLAGS.samples_per_epoch))
            samples_per_epoch = FLAGS.samples_per_epoch

        history = model.fit_generator(
            batch_generator(X=X_train, Y=y_train, label='train set', num_epochs=FLAGS.epochs, flip_images=True,
                            batch_size=FLAGS.batch_size,
                            output_shape=output_shape),
            nb_epoch=FLAGS.epochs,
            samples_per_epoch=samples_per_epoch,
            validation_data=None,
            verbose=2)

    elif train_mode == 7:
        output_shape = (40, 80, 3)
        X_train, y_train, X_val, y_val = load_track_csv()

        print('population: ', len(X_train))

        # train model
        clf = BasicELU()
        model = clf.get_model(input_shape=output_shape, output_shape=output_shape, use_weights=FLAGS.use_weights, dropout_prob=FLAGS.dropout_prob)

        samples_per_epoch = len(X_train)
        if FLAGS.samples_per_epoch is not None:
            print('overriding samples per epoch from {} to {}'.format(samples_per_epoch, FLAGS.samples_per_epoch))
            samples_per_epoch = FLAGS.samples_per_epoch

        history = model.fit_generator(
            batch_generator(X=X_train, Y=y_train, label='train set', num_epochs=FLAGS.epochs, flip_images=True,
                            batch_size=FLAGS.batch_size,
                            output_shape=output_shape),
            nb_epoch=FLAGS.epochs,
            samples_per_epoch=samples_per_epoch,
            validation_data=None,
            verbose=2)

    elif train_mode == 8:
        output_shape = (80, 160, 3)
        X_train, y_train, X_val, y_val = load_track_data(output_shape=output_shape[0:2], repickle=FLAGS.repickle)
        img_rows, img_cols = X_train.shape[1], X_train.shape[2]
        output_shape = (img_rows, img_cols, 3)

        print('X_train shape: ', X_train.shape)
        print('X_val shape:   ', X_val.shape)

        # train model
        clf = Udacity()
        model = clf.get_model(input_shape=output_shape, output_shape=output_shape, learning_rate=FLAGS.lr)
        history = model.fit(X_train, y_train, batch_size=FLAGS.batch_size, nb_epoch=FLAGS.epochs,
                            validation_data=(X_val, y_val))

        # predict 3 images and compare accuracy
        predict_images(model)

    print(history.history)
    clf.save()


    # move_training_images(clf)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
