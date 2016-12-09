# required in order to execute this file from project root.
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from zimpy.datasets.german_traffic_signs import GermanTrafficSignDataset

data = GermanTrafficSignDataset()
data.configure(one_hot=True, train_validate_split_percentage=0)

X_train = data.train_orig
y_train = data.train_labels

X_train = data.normalize_data(X_train)
X_train = X_train.astype('float32')

batch_size = 128
nb_classes = data.num_classes
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 32, 32

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

# If Theano backend, input_shape is different so let's take care of that first
# if K.image_dim_ordering() == 'th':
#     X_train = X_train.reshape(X_train.shape[0], nb_channels, img_rows, img_cols)
#     input_shape = (nb_channels, img_rows, img_cols)
# else:
#     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, nb_channels)
#     input_shape = (img_rows, img_cols, nb_channels)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, nb_channels)
input_shape = (img_rows, img_cols, nb_channels)

# build the model
model = Sequential(name='input')
model.add(Convolution2D(16, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(dropout_p_1))
model.add(Flatten())
model.add(Dense(hidden_layer_neurons, name='hidden1'))
model.add(Activation('relu'))
# model.add(Dropout(dropout_p_2))
model.add(Dense(data.num_classes))
model.add(Activation('softmax', name='output'))

# print information about the model itself
model.summary()

# Compile and train the model.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Split some of the training data into a validation dataset.
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=832224)

history = model.fit(X_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_val, y_val))

score = model.evaluate(X_val, y_val, verbose=1)
print('Validation (loss, accuracy): (%.3f, %.3f)' % (score[0], score[1]))

# STOP: Do not change the tests below. Your implementation should pass these tests.
assert (history.history['val_acc'][-1] > 0.9), "The validation accuracy is: %.3f" % history.history['val_acc'][-1]


X_test = data.test_orig
y_test = data.test_labels
X_test = X_test.astype('float32')
X_test /= 255
X_test -= 0.5

loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)
print()


X_pred = data.predict_orig
y_pred = data.predict_labels
X_pred = X_pred.astype('float32')
X_pred /= 255
X_pred -= 0.5

loss, accuracy = model.evaluate(X_pred, y_pred)

print('pred loss: ', loss)
print('pred accuracy: ', accuracy)
print()
