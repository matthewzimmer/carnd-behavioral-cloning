import os
import csv
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

from training import TrainTrackA, CommaAI, MyComma, SimpleConvnet
from zimpy.camera_preprocessor import preprocess_image, predict_images
from zimpy.serializers.trained_data_serializer import TrainedDataSerializer

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('network_arch', 'commaai', "The network architecture to train on.")
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")
flags.DEFINE_integer('samples_per_epoch', 256, "The number of samples per epoch during training.")
flags.DEFINE_integer('algo_mode', 1, "The algorithm to train against.")
flags.DEFINE_boolean('repickle', True, "Whether to regenerage the train.p file of training camera images.")

train_samples_seen = []
X_train, y_train, X_val, y_val = None, None, None, None
img_rows, img_cols = None, None


def gen_train(batch_size=16):
	global train_samples_seen, X_train, y_train

	num_training = len(X_train)
	_index_in_epoch = 0
	start_i = 0
	while True:
		# for i in range(math.floor(X_train.shape[0] / (1. * batch_size))):  # 100 * 32 = 3200 -> # of training samples
		# if i % 25 == 0:
		#     print("train i = " + str(i))
		# start_i = i * batch_size
		# end_i = (i + 1) * batch_size
		# train_samples_seen += list(range(start_i, end_i))
		# print('   train sample range: ', range(start_i, end_i))
		# yield X_train[start_i:end_i], y_train[start_i:end_i]

		for i in range(100 * batch_size):  # 100 * 16 = 1600 -> # of training samples
			_index_in_epoch += batch_size
			if _index_in_epoch > num_training:
				# Shuffle the data
				perm = np.arange(num_training)
				np.random.shuffle(perm)
				X_train = X_train[perm]
				y_train = y_train[perm]

				# Start next epoch
				start_i = 0
				_index_in_epoch = batch_size
				assert batch_size <= num_training
			end_i = _index_in_epoch
			print('  yielding train items in range {}'.format(range(start_i, end_i)))
			yield X_train[start_i:end_i], y_train[start_i:end_i]


def gen_val(batch_size=16):
	global X_train, y_train
	while True:
		for i in range(math.floor(X_train.shape[0] / (1. * batch_size))):  # 100 * 32 = 3200 -> # of validation samples
			if i % 25 == 0:
				print("validate i = " + str(i))
			start_i = i * batch_size
			end_i = (i + 1) * batch_size
			yield X_val[start_i:end_i], y_val[start_i:end_i]


def load_track_data(output_shape, repickle=False):
	pickle_file = os.path.join(os.path.dirname(__file__), 'train.p')
	if repickle == True or not os.path.isfile(pickle_file):
		X_train, y_train = [], []
		train_data_paths = []
		# train_data_paths.append('data/training/1')
		# train_data_paths.append('data/training/center_line')
		train_data_paths.append('.')
		# train_data_paths.append('data/training/2')
		# train_data_paths.append('data/training/3')
		# train_data_paths.append('data/training/4')
		# train_data_paths.append('data/training/5')
		# train_data_paths.append('data/training/6')
		# train_data_paths.append('data/training/7')
		# train_data_paths.append('data/training/8')

		for train_data_path in train_data_paths:
			drive_log_path = train_data_path + '/driving_log.csv'
			if os.path.isfile(drive_log_path):
				with open(train_data_path + '/driving_log.csv', 'r') as drive_logs:
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

							orig_image = misc.imread(image_path)
							out_image = preprocess_image(orig_image, output_shape)
							X_train.append(out_image)
							y_train.append(steering_angle)
							# if not math.isclose(steering_angle, 0.0) and random.random() < 0.5:
							# if not math.isclose(steering_angle, 0.0):
							if steering_angle > 0.0 or steering_angle < 0.0:
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
		# clf = CommaAI()
		# clf = MyVGG16()
		clf = MyComma()
		model = clf.get_model(input_shape=output_shape, output_shape=output_shape)
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
		output_shape = (int(160 / 2), int(320 / 2), 3)
		X_train, y_train, X_val, y_val = load_track_data(output_shape=output_shape[0:2], repickle=FLAGS.repickle)

		print('X_train shape: ', X_train.shape)
		print('X_val shape:   ', X_val.shape)

		# train model
		clf = CommaAI()
		model = clf.get_model(input_shape=output_shape, output_shape=output_shape)

		history = model.fit_generator(gen_train(FLAGS.batch_size),
		                              nb_epoch=FLAGS.epochs,
		                              samples_per_epoch=len(X_train),
		                              nb_val_samples=len(X_val),
		                              validation_data=gen_val(FLAGS.batch_size),
		                              verbose=2)

	print(history.history)
	clf.save()


# parses flags and calls the `main` function above
if __name__ == '__main__':
	tf.app.run()
