# required in order to execute this file from project root.
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# TODO: Re-construct the network and add dropout after the pooling layer.
# TODO: Compile and train the model.

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

# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)

# TODO: Re-construct the network and add a convolutional layer before the first fully-connected layer.
model = Sequential(name='input')
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, name='hidden1'))
model.add(Activation('relu'))
model.add(Dense(data.num_classes))
model.add(Activation('softmax', name='output'))

model.summary()

# TODO: Compile and train the model.
batch_size = 128
nb_classes = data.num_classes
nb_epoch = 2

model.compile(optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['accuracy'])

# TODO: Split some of the training data into a validation dataset.
X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.15,
            random_state=832224)

# TODO: Compile and train the model to measure validation accuracy.
history = model.fit(X_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_val, y_val))

# score = model.evaluate(X_val, y_val, verbose=1)
# print('Validation (loss, accuracy): (%.3f, %.3f)' % (score[0], score[1]))

# STOP: Do not change the tests below. Your implementation should pass these tests.
assert(history.history['val_acc'][0] > 0.9), "The validation accuracy is: %.3f" % history.history['val_acc'][0]