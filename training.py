# Tips from AlexNet paper at
# https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

# TODO: Downsample training images to 256x256
#
#       Given a rectangular image, we first rescaled the image such that the shorter side was of length 256, and then
#       cropped out the central 256×256 patch from the resulting image. We did not pre-process the images
#       in any other way, except for subtracting the mean activity over the training set from each pixel. So
#       we trained our network on the (centered) raw RGB values of the pixels.

import numpy as np
import json
import uuid
import os

from keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, ZeroPadding2D, Lambda, ELU, BatchNormalization
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam


class BaseNetwork:
    def __init__(self):
        self.uuid = uuid.uuid4()
        self.model = None
        self.weights = None
        self.__configured = True

    def fit(self, X_train, y_train, nb_epoch=12, batch_size=128, validation_data=None, shuffle=True):
        raise NotImplementedError

    def save(self):
        print('Saved {} model.'.format(self.__class__.__name__))
        self.__persist()

    def serialize(self, data={}):
        if self.__configured:
            return {
                **data,
                **{
                    'model': self.model.to_json(),
                    'weights': self.model.get_weights()
                }
            }
        else:
            return data

    def restore(self, optimizer="adam", loss="mse"):
        save_dir = os.path.join(os.path.dirname(__file__), 'data', 'trained')
        weights_save_path = os.path.join(save_dir, '{}_{}.h5'.format('model', self.__class__.__name__))
        model_save_path = os.path.join(save_dir, '{}_{}.json'.format('model', self.__class__.__name__))
        model = None
        if os.path.exists(model_save_path):
            with open(model_save_path, 'r') as jfile:
                the_json = json.load(jfile)
                print(json.loads(the_json))
                model = model_from_json(the_json)
                model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            if os.path.exists(weights_save_path):
                model.load_weights(weights_save_path)
        return model

    def __persist(self):
        save_dir = os.path.join(os.path.dirname(__file__), 'data', 'trained')
        weights_save_path = os.path.join(save_dir, '{}_{}.h5'.format('model', self.__class__.__name__))
        model_save_path = os.path.join(save_dir, '{}_{}.json'.format('model', self.__class__.__name__))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.model.save_weights(weights_save_path)
        # self.model.save_weights('model.h5')

        model_json = self.model.to_json()
        # with open('model.json', 'w') as outfile:
        # 	json.dump(model_json, outfile)
        with open(model_save_path, 'w') as outfile:
            json.dump(model_json, outfile)

    def __str__(self):
        results = [self.model.summary()]
        return '\n'.join(results)


class TrainTrackA(BaseNetwork):
    def fit(self, X_train, y_train, nb_epoch=12, batch_size=128, validation_data=None, shuffle=True):
        nb_classes = len(np.unique(y_train))

        # input image dimensions
        img_rows, img_cols = X_train.shape[1], X_train.shape[2]

        # number of convolutional filters to use
        nb_filters = 32

        # number of channels for our input image
        nb_channels = 3

        # convolution kernel size
        kernel_size = (5, 5)

        # size of pooling area for max pooling
        pool_size = (2, 2)

        # number of neurons for our hidden layer
        hidden_layer_neurons = 128

        # A float between 0 and 1. Fraction of the input units to drop.
        dropout_p_1, dropout_p_2 = 0.5, 0.5

        # train rows/cols

        # TODO: rescale so shorter size is length 256
        # TODO: crop central 256x256 patch from that image
        # img_rows, img_cols = (256, 256)
        # X_train.reshape(X_train.shape[0], img_rows, img_cols, X_train.shape[-1])

        input_shape = (img_rows, img_cols, nb_channels)

        self.model = Sequential(name='input')
        self.model.add(Convolution2D(16, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=pool_size))
        self.model.add(Dropout(dropout_p_1))
        self.model.add(Flatten())
        self.model.add(Dense(hidden_layer_neurons, name='hidden1'))
        self.model.add(Activation('relu'))
        # self.model.add(Dropout(dropout_p_2))
        self.model.add(Dense(10))
        self.model.add(Dense(1))
        self.model.add(Activation('softmax', name='output'))

        # print information about the model itself
        self.model.summary()

        # Compile and train the model.
        self.model.compile(optimizer='adam',
                           loss='mse',
                           metrics=['accuracy'])

        history = self.model.fit(X_train, y_train,
                                 batch_size=batch_size, nb_epoch=nb_epoch,
                                 verbose=1, validation_data=validation_data)

        print(history.history)

        # score = self.model.evaluate(X_val, y_val, verbose=1)
        # print('Validation (loss, accuracy): (%.3f, %.3f)' % (score[0], score[1]))

        # STOP: Do not change the tests below. Your implementation should pass these tests.
        # assert (history.history['val_acc'][-1] > 0.9), "The validation accuracy is: %.3f" % history.history['val_acc'][-1]


class TrainTrackB(BaseNetwork):
    def fit(self, X_train, y_train, nb_epoch=12, batch_size=128, validation_data=None, shuffle=True):
        nb_classes = len(np.unique(y_train))

        # input image dimensions
        img_rows, img_cols = X_train.shape[1], X_train.shape[2]

        # number of convolutional filters to use
        nb_filters = 32

        # number of channels for our input image
        nb_channels = 3

        # convolution kernel size
        kernel_size = (5, 5)

        # size of pooling area for max pooling
        pool_size = (2, 2)

        # number of neurons for our hidden layer
        hidden_layer_neurons = 128

        # A float between 0 and 1. Fraction of the input units to drop.
        dropout_p_1, dropout_p_2 = 0.5, 0.5

        # train rows/cols

        # TODO: rescale so shorter size is length 256
        # TODO: crop central 256x256 patch from that image
        # img_rows, img_cols = (256, 256)
        # X_train.reshape(X_train.shape[0], img_rows, img_cols, X_train.shape[-1])

        input_shape = (img_rows, img_cols, nb_channels)

        self.model = Sequential(name='input')
        self.model.add(Convolution2D(16, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))

        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=pool_size))
        self.model.add(Dropout(dropout_p_1))
        self.model.add(Flatten())
        self.model.add(Dense(hidden_layer_neurons, name='hidden1'))
        self.model.add(Activation('relu'))
        # self.model.add(Dropout(dropout_p_2))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax', name='output'))

        # print information about the model itself
        self.model.summary()

        # Compile and train the model.
        self.model.compile(optimizer='adam',
                           loss='mse',
                           metrics=['accuracy'])

        history = self.model.fit(X_train, y_train,
                                 batch_size=batch_size, nb_epoch=nb_epoch,
                                 verbose=1, validation_data=validation_data)

        print(history.history)

        # score = self.model.evaluate(X_val, y_val, verbose=1)
        # print('Validation (loss, accuracy): (%.3f, %.3f)' % (score[0], score[1]))

        # STOP: Do not change the tests below. Your implementation should pass these tests.
        # assert (history.history['val_acc'][-1] > 0.9), "The validation accuracy is: %.3f" % history.history['val_acc'][-1]


class SimpleConvnet(BaseNetwork):
    def get_model(self, input_shape, output_shape, learning_rate=0.001, dropout_prob=0.1, activation='relu', use_weights=False):
        model = Sequential(name='input')
        model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(128, 3, 3, border_mode="same", activation=activation))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1024, activation=activation))
        model.add(Dense(1024, activation=activation))
        model.add(Dense(512, activation=activation))
        model.add(Dense(256, activation=activation))
        model.add(Dropout(0.5))
        # model.add(Activation('relu'))
        model.add(Dense(1))

        # print information about the model itself
        model.summary()

        # Compile and train the model.
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        # history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose, validation_data=validation_data)
        # print(history.history)

        self.model = model

        return self.model


class CommaAI(BaseNetwork):
    """
    Downloaded from https://github.com/commaai/research/blob/master/train_steering_model.py
    """

    def get_model(self, input_shape, output_shape, learning_rate=0.0001, use_weights=False):
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.,
                         input_shape=input_shape,
                         output_shape=output_shape))
        model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(Flatten())
        model.add(Dropout(.2))
        model.add(ELU())
        model.add(Dense(512))
        model.add(Dropout(.5))
        model.add(Dense(512))
        model.add(Dropout(.5))
        model.add(Dense(128))
        model.add(Dropout(.5))
        model.add(ELU())
        model.add(Dense(1))

        # print information about the model itself
        model.summary()

        # Compile and train the model.
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        # history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose, validation_data=validation_data)
        # print(history.history)

        self.model = model

        return self.model


class Basic(BaseNetwork):
    def get_model(self, input_shape, output_shape, learning_rate=0.001, dropout_prob=0.1, activation='relu', use_weights=False):
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
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
        self.model = model
        model.summary()
        return model


class BasicELU(BaseNetwork):
    def get_model(self, input_shape, output_shape, learning_rate=0.001, dropout_prob=0.5, activation='elu', use_weights=False):
        model = None

        optimizer = Adam(lr=learning_rate)
        loss = 'mean_squared_error'

        if use_weights:
            model = self.restore(optimizer=optimizer, loss=loss)
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
            # model.add(MaxPooling2D(pool_size=(2, 2)))
            # model.add(Convolution2D(128, 3, 3, border_mode='same', activation=activation))
            model.add(Flatten())
            model.add(Dropout(dropout_prob))
            model.add(Dense(1024, activation=activation))
            model.add(Dropout(dropout_prob))
            model.add(Dense(100, activation=activation))
            model.add(Dense(50, activation=activation))
            model.add(Dense(10, activation=activation))
            model.add(Dense(1, init='normal'))

        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        self.model = model
        model.summary()
        return model


class Nvidia(BaseNetwork):
    def get_model(self, input_shape, output_shape, learning_rate=0.0001, dropout_prob=0.5, activation='elu', use_weights=False):
        model = None

        optimizer = Adam(lr=learning_rate)
        loss = 'mse'
        if use_weights:
            model = self.restore(optimizer=optimizer, loss=loss)
        if model is None:
            model = Sequential()
            model.add(Lambda(lambda x: x / 127.5 - 1.,
                             input_shape=input_shape,
                             output_shape=output_shape))
            model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", activation=activation))
            model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", activation=activation))
            model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", activation=activation))
            model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", activation=activation))
            model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", activation=activation))
            model.add(Flatten())
            model.add(ELU())
            model.add(Dense(1164))
            model.add(Dropout(dropout_prob))
            model.add(Dense(100))
            model.add(Dropout(dropout_prob))
            model.add(Dense(50))
            model.add(Dropout(dropout_prob))
            model.add(ELU())
            model.add(Dense(1))
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        model.summary()
        self.model = model

        return self.model


class Udacity(BaseNetwork):
    def get_model(self, input_shape=None, output_shape=None, learning_rate=0.0001, dropout_prob=0.5, activation='relu', use_weights=False):
        self.model = self.restore(optimizer=Adam(lr=learning_rate), loss='mse')
        if self.model is None:
            clf = Nvidia()
            # clf = CommaAI()
            self.model = clf.get_model(input_shape, output_shape, learning_rate, use_weights=False)
        self.model.summary()
        return self.model


class SimpleConvnet(BaseNetwork):
    def get_model(self, input_shape, output_shape, learning_rate=0.0001, dropout_prob=0.5, activation='elu', use_weights=False):
        model = None

        optimizer = Adam(lr=learning_rate)
        loss = 'msle'
        if use_weights:
            model = self.restore(optimizer=optimizer, loss=loss)
        if model is None:
            model = Sequential()
            model.add(BatchNormalization(input_shape=input_shape, axis=1))
            model.add(Convolution2D(32, 3, 3, border_mode="same", input_shape=input_shape))
            model.add(MaxPooling2D((2, 2), strides=(2, 2), activation=activation))
            model.add(Convolution2D(64, 3, 3, border_mode="same"))
            model.add(Flatten())
            model.add(Dense(128, activation=activation))
            model.add(Dense(1))

        # print information about the model itself
        model.summary()

        # Compile and train the model.
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        self.model = model

        return self.model


class MyVGG16(BaseNetwork):
    """
    Downloaded from https://github.com/commaai/research/blob/master/train_steering_model.py
    """

    def get_model(self, input_shape, output_shape, learning_rate=0.0001, dropout_prob=0.5, activation='relu', use_weights=False):
        model = Sequential()
        # model.add(Input(input_shape=input_shape, output_shape=output_shape))
        # model.add(Lambda(lambda x: x / 127.5 - 1.,
        # model.add(Lambda(lambda x: x / 1.,
        #                  input_shape=input_shape,
        #                  output_shape=output_shape))
        model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
        model.add(Convolution2D(64, 3, 3, activation=activation))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation=activation))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation=activation))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation=activation))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation=activation))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation=activation))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation=activation))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation=activation))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation=activation))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation=activation))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation=activation))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation=activation))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation=activation))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation=activation))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation=activation))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))
        model.add(Dropout(.5))
        model.add(ELU())
        model.add(Dense(1))

        # print information about the model itself
        model.summary()

        # Compile and train the model.
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        # history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose, validation_data=validation_data)
        # print(history.history)

        self.model = model

        return self.model


class VGG16(BaseNetwork):
    def get_model(self, input_shape, output_shape, learning_rate=0.0001, dropout_prob=0.5, activation='relu', use_weights=False):
        model = Sequential()

        # TODO: Ensure to provide proper input_shape
        model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
        model.add(Convolution2D(64, 3, 3, activation=activation))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation=activation))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation=activation))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation=activation))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation=activation))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation=activation))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation=activation))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation=activation))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation=activation))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation=activation))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation=activation))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation=activation))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation=activation))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation=activation))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation=activation))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))
        model.add(Dropout(.5))
        model.add(ELU())
        model.add(Dense(1))

        # if weights_path:
        #     model.load_weights(weights_path)

        # return model

        # print information about the model itself
        model.summary()

        # Compile and train the model.
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        # history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=validation_data)

        # print(history.history)

        self.model = model

        return self.model
