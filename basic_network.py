import numpy as np
import pandas as pd
import json
import uuid
import os
import random
import cv2
import math

from scipy import misc
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, ZeroPadding2D, Lambda, ELU, \
    BatchNormalization
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from zimpy.plot.image_plotter import ImagePlotter


class RecordingMeasurement:
    """
    A representation of a vehicle's state at a point in time while driving
    around a track during recording.

    Features available are:

        left_camera_view   - An image taken by the LEFT camera.
        center_camera_view - An image taken by the CENTER camera.
        right_camera_view  - An image taken by the RIGHT camera.
        steering_angle     - A normalized steering angle in the range -1 to 1.
        speed              - The speed in which the vehicle was traveling at measurement time.


    This class serves the following purposes:

      1. Provides convenience getter methods for left, center and camera images.
         In an effort to reduce memory footprint, they're essentially designed
         to lazily instantiate (once) the actual image array at the time the
         method is invoked.

      2. Strips whitespace off the left, center, and right camera image paths.

      3. Casts the original absolute path of each camera image to a relative path.
         This adds reassurance the image will load on any computer.

      4. Provides a convenient #is_valid_measurment method which encapsulates
         pertinent logic to ensure data quality is satisfactory.

    """

    def __init__(self, measurement_data):
        self.measurement_data = measurement_data

        self.steering_angle = round(float(measurement_data['steering']), 4)
        self.speed = round(float(measurement_data['speed']), 4)

        l = measurement_data['left'].strip()
        c = measurement_data['center'].strip()
        r = measurement_data['right'].strip()

        # cast absolute path to relative path to be environment agnostic
        l, c, r = [('./IMG/' + os.path.split(file_path)[1]) for file_path in (l, c, r)]

        self.left_camera_view_path = l
        self.center_camera_view_path = c
        self.right_camera_view_path = r

    def is_valid_measurement(self):
        """
        Return true if the original center image is available to load.
        """
        return os.path.isfile(self.center_camera_view_path)

    def left_camera_view(self):
        """
        Lazily instantiates the left camera view image at first call.
        """
        if not hasattr(self, '__left_camera_view'):
            self.__left_camera_view = self.__load_image(self.left_camera_view_path)
        return self.__left_camera_view

    def center_camera_view(self):
        """
        Lazily instantiates the center camera view image at first call.
        """
        if not hasattr(self, '__center_camera_view'):
            self.__center_camera_view = self.__load_image(self.center_camera_view_path)
        return self.__center_camera_view

    def right_camera_view(self):
        """
        Lazily instantiates the right camera view image at first call.
        """
        if not hasattr(self, '__right_camera_view'):
            self.__right_camera_view = self.__load_image(self.right_camera_view_path)
        return self.__right_camera_view

    def __load_image(self, imagepath):
        image_array = None
        if os.path.isfile(imagepath):
            image_array = misc.imread(imagepath)
        else:
            print('File Not Found: {}'.format(imagepath))
        return image_array

    def __str__(self):
        results = []
        results.append('Image paths:')
        results.append('')
        results.append('     Left camera path: {}'.format(self.left_camera_view_path))
        results.append('   Center camera path: {}'.format(self.center_camera_view_path))
        results.append('    Right camera path: {}'.format(self.right_camera_view_path))
        results.append('')
        results.append('Additional features:')
        results.append('')
        results.append('   Steering angle: {}'.format(self.steering_angle))
        results.append('            Speed: {}'.format(self.speed))
        return '\n'.join(results)


def preprocess_image(image_array, output_shape=(40, 80), colorspace='yuv'):
    """
    Reminder:

    Source image shape is (160, 320, 3)

    Our preprocessing algorithm consists of the following steps:

      1. Converts BGR to YUV colorspace.

         This allows us to leverage luminance (Y channel - brightness - black and white representation),
         and chrominance (U and V - blue–luminance and red–luminance differences respectively)

      2. Crops top 31.25% portion and bottom 12.5% portion.
         The entire width of the image is preserved.

         This allows the model to generalize better to unseen roadways since we clop
         artifacts such as trees, buildings, etc. above the horizon. We also clip the
         hood from the image.

      3. Finally, I allow users of this algorithm the ability to specify the shape of the final image via
         the output_shape argument.

         Once I've cropped the image, I resize it to the specified shape using the INTER_AREA
         interpolation agorithm as it is the best choice to preserve original image features.

         See `Scaling` section in OpenCV documentation:

         http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
    """
    # convert image to another colorspace
    if colorspace == 'yuv':
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2YUV)
    elif colorspace == 'hsv':
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    elif colorspace == 'rgb':
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # [y1:y2, x1:x2]
    #
    # crops top 31.25% portion and bottom 12.5% portion
    #
    # The entire width of the image is preserved
    image_array = image_array[50:140, 0:320]

    # Let's blur the image to smooth out some of the artifacts
    kernel_size = 5  # Must be an odd number (3, 5, 7...)
    image_array = cv2.GaussianBlur(image_array, (kernel_size, kernel_size), 0)

    # resize image to output_shape
    image_array = cv2.resize(image_array, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_AREA)

    return image_array


class Track1TrainingDataset:
    """
    Parses driving_log.csv and constructs training and test datasets corresponding to
    measurements taken at various points in time while recording on track 1.
    """

    DRIVING_LOG_PATH = './driving_log.csv'

    def __init__(self, validation_split_percentage=0.01):
        self.X_train = []
        self.X_val = []
        self.y_train = []
        self.y_val = []
        self.dataframe = None
        self.headers = []
        self.__loaded = False
        self.load(validation_split_percentage=validation_split_percentage)

    def load(self, validation_split_percentage):
        if not self.__loaded:
            X_train, y_train = [], []

            if os.path.isfile(self.DRIVING_LOG_PATH):
                df = pd.read_csv(self.DRIVING_LOG_PATH)
                headers = list(df.columns.values)
                for index, measurement_data in df.iterrows():
                    measurement = RecordingMeasurement(measurement_data=measurement_data)
                    X_train.append(measurement)
                    y_train.append(measurement.steering_angle)

            # Split some of the training data into a validation dataset
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=validation_split_percentage,
                random_state=0)

            X_train, y_train, X_val, y_val = np.array(X_train), np.array(y_train, dtype=np.float32), np.array(
                X_val), np.array(
                y_val, dtype=np.float32)

            self.X_train = X_train
            self.X_val = X_val
            self.y_train = y_train
            self.y_val = y_val
            self.dataframe = df
            self.headers = headers
            self.__loaded = True

        return self.X_train, self.y_train, self.X_val, self.y_val

    def batch_generator(self, X, Y, label, num_epochs, batch_size=32, output_shape=(160, 320), flip_images=True,
                        classifier=None, colorspace='yuv'):
        """
        A custom batch generator with the main goal of reducing memory footprint
        on computers and GPUs with limited memory space.

        Infinitely yields `batch_size` elements from the X and Y datasets.

        During batch iteration, this algorithm randomly flips the image
        and steering angle to reduce bias towards a specific steering angle/direction.
        """
        population = len(X)
        counter = 0
        _index_in_epoch = 0
        _tot_epochs = 0
        batch_size = min(batch_size, population)
        batch_count = int(math.ceil(population / batch_size))

        assert X.shape[0] == Y.shape[0], 'X and Y size must be identical.'

        print('Batch generating against the {} dataset with population {} and shape {}'.format(label, population,
                                                                                               X.shape))
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
                    measurement = X[j]
                    center_image = measurement.center_camera_view()
                    if center_image is not None:
                        image = preprocess_image(center_image, output_shape=output_shape, colorspace=colorspace)

                        # Here I throw in a random image flip to reduce bias towards
                        # a specific direction/steering angle.
                        if flip_images and random.random() > 0.5:
                            X_batch.append(np.fliplr(image))
                            y_batch.append(-steering_angle)
                        else:
                            X_batch.append(image)
                            y_batch.append(steering_angle)

                yield np.array(X_batch), np.array(y_batch)

    def __str__(self):
        results = []
        results.append('{} Stats:'.format(self.__class__.__name__))
        results.append('')
        results.append('  [Headers]')
        results.append('')
        results.append('    {}'.format(self.headers))
        results.append('')
        results.append('')
        results.append('  [Shapes]')
        results.append('')
        results.append('    training features: {}'.format(self.X_train.shape))
        results.append('    training labels: {}'.format(self.y_train.shape))
        results.append('')
        results.append('    validation features: {}'.format(self.X_val.shape))
        results.append('    validation labels: {}'.format(self.y_val.shape))
        return '\n'.join(results)


def load_dataset(validation_split_percentage=0.05):
    dataset = Track1TrainingDataset(validation_split_percentage=validation_split_percentage)
    print(dataset)
    print(dataset.dataframe.head(n=5))
    return dataset


def visualize_dataset(dataset):
    dataset.dataframe.plot.hist(alpha=0.5)
    dataset.dataframe['steering'].plot.hist(alpha=0.5)
    dataset.dataframe['steering'].plot(alpha=0.5)


def visualize_features(dataset):
    perm = np.arange(len(dataset.X_train))
    np.random.shuffle(perm)
    output_shape = (40, 80, 3)
    generator = dataset.batch_generator(
        colorspace='yuv',
        X=dataset.X_train[0:10],
        Y=dataset.y_train[0:10],
        output_shape=output_shape,
        label='batch feature exploration',
        num_epochs=1,
        batch_size=10
    )

    # Grab the first 10 items from the training set and
    X_batch, y_batch = next(generator)
    print(X_batch.shape)
    print(y_batch.shape)

    # Cast to string so they render nicely in graph
    y_batch = [str(x) for x in y_batch]

    ImagePlotter.plot_images(X_batch, y_batch, rows=2, columns=5)
    ImagePlotter.plot_images(X_batch[:, :, :, 0], y_batch, rows=2, columns=5)
    ImagePlotter.plot_images(X_batch[:, :, :, 1], y_batch, rows=2, columns=5)
    ImagePlotter.plot_images(X_batch[:, :, :, 2], y_batch, rows=2, columns=5)


class BaseNetwork:
    WEIGHTS_FILE_NAME = 'model_final.h5'
    MODEL_FILE_NAME = 'model_final.json'

    def __init__(self):
        self.uuid = uuid.uuid4()
        self.model = None
        self.weights = None

    def fit(self, model, batch_generator, X_train, y_train, X_val, y_val, nb_epoch=2, batch_size=32,
            samples_per_epoch=None, output_shape=(160, 320, 3)):
        raise NotImplementedError

    def build_model(self, input_shape, output_shape, learning_rate=0.001, dropout_prob=0.1, activation='relu'):
        raise NotImplementedError

    def save(self):
        print('Saved {} model.'.format(self.__class__.__name__))
        self.__persist()

    def restore(self):
        model = None
        if os.path.exists(self.MODEL_FILE_NAME):
            with open(self.MODEL_FILE_NAME, 'r') as jfile:
                the_json = json.load(jfile)
                print(json.loads(the_json))
                model = model_from_json(the_json)
            if os.path.exists(self.WEIGHTS_FILE_NAME):
                model.load_weights(self.WEIGHTS_FILE_NAME)
        return model


    def __persist(self):
        save_dir = os.path.join(os.path.dirname(__file__))
        weights_save_path = os.path.join(save_dir, self.WEIGHTS_FILE_NAME)
        model_save_path = os.path.join(save_dir, self.MODEL_FILE_NAME)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.model.save_weights(weights_save_path)
        with open(model_save_path, 'w') as outfile:
            json.dump(self.model.to_json(), outfile)

    def __str__(self):
        results = []
        if self.model is not None:
            results.append(self.model.summary())
        return '\n'.join(results)


class Track1(BaseNetwork):
    def fit(self, model, batch_generator, X_train, y_train, X_val, y_val, nb_epoch=2, batch_size=32,
            samples_per_epoch=None, output_shape=(40, 80, 3)):
        # Fit the model leveraging the custom
        # batch generator baked into the
        # dataset itself.
        history = model.fit_generator(
            batch_generator(
                X=X_train,
                Y=y_train,
                label='train set',
                num_epochs=nb_epoch,
                batch_size=batch_size,
                output_shape=output_shape,
                classifier=self
            ),
            nb_epoch=nb_epoch,
            samples_per_epoch=len(X_train),
            nb_val_samples=len(X_val),
            verbose=2,
            validation_data=batch_generator(
                X=X_val,
                Y=y_val,
                label='validation set',
                num_epochs=nb_epoch,
                batch_size=batch_size,
                output_shape=output_shape
            )
        )

        print(history.history)
        self.save()

    def build_model(self, input_shape, output_shape, learning_rate=0.001, dropout_prob=0.1, activation='relu', use_weights=False):
        """
        Inital zero-mean normalization input layer.
        A 4-layer deep neural network with 4 fully connected layers at the top.
        ReLU activation used on each convolution layer.
        Dropout of 10% (default) used after initially flattening after convolution layers.
        Dropout of 10% (default) used after first fully connected layer.

        Adam optimizer with 0.001 learning rate (default) used in this network.
        Mean squared error loss function was used since this is a regression problem and MSE is
        quite common and robust for regression analysis.
        """
        model = None
        if use_weights:
            model = self.restore()
        if model is None:
            model = Sequential()
            model.add(Lambda(lambda x: x / 255 - 0.5,
                             input_shape=input_shape,
                             output_shape=output_shape))
            model.add(Convolution2D(24, 5, 5, border_mode='valid', activation=activation))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Convolution2D(36, 5, 5, border_mode='valid', activation=activation))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Convolution2D(48, 5, 5, border_mode='same', activation=activation))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Convolution2D(64, 3, 3, border_mode='same', activation=activation))
            model.add(Flatten())
            model.add(Dropout(dropout_prob))
            model.add(Dense(1024, activation=activation))
            model.add(Dropout(dropout_prob))
            model.add(Dense(100, activation=activation))
            model.add(Dense(50, activation=activation))
            model.add(Dense(10, activation=activation))
            model.add(Dense(1, init='normal'))

        optimizer = Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.model = model
        model.summary()
        return model


def train_network(nb_epoch=2, batch_size=32, validation_split_percentage=0.05, output_shape=(40, 80, 3),
                  learning_rate=0.001, dropout_prob=0.1, activation='relu', use_weighs=False):
    dataset = load_dataset(validation_split_percentage=validation_split_percentage)
    # visualize_dataset(dataset)
    # visualize_features(dataset)

    if len(dataset.X_train) > 0:
        print('Center camera view shape:\n\n{}\n'.format(dataset.X_train[0].center_camera_view().shape))
        print(dataset.X_train[0])

    clf = Track1()
    model = clf.build_model(
        input_shape=output_shape,
        output_shape=output_shape,
        learning_rate=learning_rate,
        dropout_prob=dropout_prob,
        activation=activation,
        use_weights=use_weighs
    )

    clf.fit(
        model=model,
        batch_generator=dataset.batch_generator,
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_val=dataset.X_val,
        y_val=dataset.y_val,
        nb_epoch=nb_epoch,
        batch_size=batch_size
    )
