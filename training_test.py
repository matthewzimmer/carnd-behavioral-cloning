import os
import csv
import tensorflow as tf
import numpy as np
from keras.applications import VGG16

from scipy import misc
from sklearn.model_selection import train_test_split

from training import TrainTrackA, CommaAI

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('network_arch', 'commaai', "The network architecture to train on.")
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")


def load_track_data():
    X_train, y_train = [], []

    train_data_paths = []
    train_data_paths.append('data/training/1')
    # train_data_paths.append('data/training/2')

    for train_data_path in train_data_paths:
        drive_log_path = train_data_path+'/drive_log.csv'
        if os.path.isfile(drive_log_path):
            with open(train_data_path+'/drive_log.csv', 'r') as drive_logs:
                has_header = csv.Sniffer().has_header(drive_logs.read(1024))
                drive_logs.seek(0)  # rewind
                incsv = csv.reader(drive_logs)
                if has_header:
                    next(incsv)  # skip header row
                plots = csv.reader(drive_logs, delimiter=',')
                for row in plots:
                    X_train.append(misc.imread(row[0]))
                    y_train.append(float(row[3]))


    # Split some of the training data into a validation dataset.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=0)

    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)

def main(_):
    X_train, y_train, X_val, y_val = load_track_data()

    if FLAGS.network_arch == 'commaai':
        model = CommaAI()
    elif FLAGS.network_arch == 'vgg16':
        model = VGG16(include_top=True, weights='imagenet',
                      input_tensor=None, input_shape=None)
    else:
        raise NotImplementedError

    # train model
    model.fit(X_train, y_train, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val), shuffle=True)
    model.save()


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
