# required in order to execute this file from project root.
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

# TODO: Implement load the data here.
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import np_utils

from zimpy.datasets.german_traffic_signs import GermanTrafficSignDataset

data = GermanTrafficSignDataset()
data.configure(one_hot=True, train_validate_split_percentage=0)
print(data)
X_train = data.train_orig
y_train = data.train_labels
nb_classes = data.num_classes

# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)

# print(data)

# STOP: Do not change the tests below. Your implementation should pass these tests.
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."



# TODO: Implement data normalization here.
X_train = data.normalize_data(X_train)

# STOP: Do not change the tests below. Your implementation should pass these tests.
assert(round(np.mean(X_train)) == 0), "The mean of the input data is: %f" % np.mean(X_train)
assert(np.min(X_train) == -0.5 and np.max(X_train) == 0.5), "The range of the input data is: %.1f to %.1f" % (np.min(X_train), np.max(X_train))


# TODO: Build a two-layer feedforward neural network with Keras here.
# as first layer in a sequential model:
model = Sequential(name='input')
model.add(Dense(128, input_dim=32*32*3, name='hidden1'))
# model.add(Flatten())
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax', name='output'))

model.summary()

# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

# history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# STOP: Do not change the tests below. Your implementation should pass these tests.
assert(model.get_layer(name="hidden1").input_shape == (None, 32*32*3)), "The input shape is: %s" % model.get_layer(name="hidden1").input_shape
assert(model.get_layer(name="output").output_shape == (None, 43)), "The output shape is: %s" % model.get_layer(name="output").output_shape

# TODO: Compile and train the model here.
batch_size = 128
nb_classes = data.num_classes
nb_epoch = 2

X_train = X_train.reshape(X_train.shape[0], 32*32*3)
# X_train = X_train.astype('float32')

model.compile(optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1)

# STOP: Do not change the tests below. Your implementation should pass these tests.
assert(history.history['acc'][0] > 0.5), "The training accuracy was: %.3f" % history.history['acc']




from sklearn.model_selection import train_test_split

# TODO: Split some of the training data into a validation dataset.
X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.25,
            random_state=832224)

# TODO: Compile and train the model to measure validation accuracy.
history = model.fit(X_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_val, y_val))

score = model.evaluate(X_val, y_val, verbose=1)

# STOP: Do not change the tests below. Your implementation should pass these tests.
assert(round(X_train.shape[0] / float(X_val.shape[0])) == 3), "The training set is %.3f times larger than the validation set." % X_train.shape[0] / float(X_val.shape[0])
assert(history.history['val_acc'][0] > 0.6), "The validation accuracy is: %.3f" % history.history['val_acc'][0]