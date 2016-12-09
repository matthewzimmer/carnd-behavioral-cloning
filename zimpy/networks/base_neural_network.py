from datetime import datetime

import os
import uuid

import tensorflow as tf
import numpy as np

from serializers.trained_data_serializer import TrainedDataSerializer


class HyperParametersContext:
    def __init__(
            self,
            start_learning_rate=0.2,
            end_learning_rate=0.2,
            epochs=15,
            batch_size=20,
            exponential_decay_config=None,
            required_accuracy_improvement=100
    ):
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.exponential_decay_config = exponential_decay_config

        # Stop optimization if no improvement found in this many iterations.
        self.required_accuracy_improvement = required_accuracy_improvement

    def __str__(self):
        result = []
        result.append(' ')
        for k, v in self.__dict__.items():
            result.append('  {} - {}'.format(k, v))
        result.append(' ')
        return '\n'.join(result)


class ConfigurationContext:
    OPTIMIZER_TYPE_GRADIENT_DESCENT = 'tf.train.GradientDescentOptimizer'
    OPTIMIZER_TYPE_ADAGRAD = 'tf.train.AdagradOptimizer'

    def __init__(self, dataset, optimizer_type=OPTIMIZER_TYPE_GRADIENT_DESCENT, hyper_parameters=None):
        """
        :param dataset: An instance of datasets.GermanTrafficSignDataset
        :param hyper_parameters: An instance of HyperParametersContext
        """
        self.data = dataset
        self.optimizer_type = optimizer_type
        if hyper_parameters is None:
            self.hyper_parameters = HyperParametersContext()
        else:
            self.hyper_parameters = hyper_parameters


class BaseNeuralNetwork:
    MINIMUM_VALIDATION_ACCURACY_CHECKPOINT_THRESHOLD = 0.85

    def __init__(self):
        self.uuid = uuid.uuid4()

        self.config = None
        self.__configured = False

        self.loss = None
        self.weights = None
        self.biases = None

        self.train_predictions = None
        self.test_predictions = None
        self.validate_predictions = None

        self.train_accuracy = None
        self.validate_accuracy = None
        self.test_accuracy = None

        # Best validation accuracy seen so far.
        self.best_validation_accuracy = 0.0

        # Iteration-number for last improvement to validation accuracy.
        self.last_improvement = 0

        # In order to save the variables of the neural network, we now create a so-called Saver-object which is used
        # for storing and retrieving all the variables of the TensorFlow graph. Nothing is actually saved at this
        # point, which will be done further below in the optimize()-function.
        self.saver = None

        # Directory where all trained models will be stored
        self.save_dir = os.path.join(os.path.dirname(__file__), '..', 'trained_models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def configure(self, configuration_context):
        """
        Principle entry point into all BaseNeuralNetworks.

        :param configuration_context: An instance of ConfigurationContext
        :return:
        """
        self.config = configuration_context
        self.__configured = True

    def generate(self):
        """
        Principle entry point into the network.

        Invokes the following methods in this order:

          1. #fit           - Does the heavy lifting of training the network
          2. #validate      - Checks the accuracy of the model against the validation set
          3. #predict       - Checks the accuracy of the model against the test set
          4. #serialize     - Serialize the entire dataset.
          5. #persist       - If a validation accuracy greater than MINIMUM_VALIDATION_ACCURACY_CHECKPOINT_THRESHOLD


        :return: None
        """
        if self.__configured:
            self.__say_log('Model fit started.')

            [self.__with_time(op['label'], op['callback']) for op in [
                {'label': 'FIT MODEL', 'callback': self.fit},
                {'label': 'SERIALIZE TRAINED MODEL', 'callback': self.serialize},
                {'label': 'PERSIST SERIALIZED TRAINED MODEL', 'callback': self.__persist}
            ]]
            self.__say_log('Model fit complete!')

            if self.__accuracy_satisfies_minimum_requirements(self.best_validation_accuracy):
                messages = []
                messages.append('The best validation accuracy achieved was {:.002f}% at iteration {}.'.format(
                    (self.best_validation_accuracy * 100), self.last_improvement))
                messages.append('Network serialized to the data directory.')
                messages.append(
                    'The most accurate validation model has been serialized to the trained models directory.')

                for msg in messages:
                    self.__say_log(msg)
            else:
                msg = 'The best validation accuracy achieved was {:.002f}% which is below the minimum requirement of {}%. No data was persisted.'.format(
                    self.best_validation_accuracy * 100,
                    str(int(self.MINIMUM_VALIDATION_ACCURACY_CHECKPOINT_THRESHOLD * 100)))
                self.__say_log(msg)

    def predict(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def save_path(self):
        """
        Save all variables of the TensorFlow graph to file.
        :return: The path to the save checkpoint file.
        """
        return os.path.join(self.save_dir, self.__generate_file_name())

    def track_loss(self, current_loss):
        """
        Keeps track of the loss at each iteration.
        :param current_loss: The value of the loss at a given point in time during training the model.
        :return:
        """
        if not hasattr(self, 'losses'):
            self.losses = []
        self.losses.append(current_loss)

    def evaluate_accuracy(self, tf_session, validation_accuracy_pct, total_iterations):
        """
        Saves the current TensorFlow model variables as they were at the time of observation if the provided accuracy is greater
        than the previously declared best validation accuracy.

        :param tf_session: A TensorFlow Session to save checkpoints against.
        :param validation_accuracy_pct: A float corresponding to a validation accuracy measurement.
        :param total_iterations: The total number of iterations taken to achieve the accuracy percentage.
        :return: Returns True if a checkpoint was saved. Otherwise False.
        """
        if self.__configured:
            if validation_accuracy_pct > self.best_validation_accuracy:

                # Update the best-known validation accuracy.
                self.best_validation_accuracy = validation_accuracy_pct

                # Set the iteration for the last improvement to current.
                self.last_improvement = total_iterations

                if self.__accuracy_satisfies_minimum_requirements(self.best_validation_accuracy):
                    if self.saver is None:
                        self.saver = tf.train.Saver()

                    # Save all variables of the TensorFlow graph to file.
                    # save_path = os.path.join(self.save_dir, self.__generate_file_name())

                    self.saver.save(sess=tf_session, save_path=self.save_path())
                    return True

                self.__say_log('{:.002f}% accuracy'.format(self.best_validation_accuracy * 100))
        return False

    def serialize(self, data={}):
        if self.__configured:
            return {
                **data,
                **{
                    'top_5_predicted_classes': self.top_5,
                    'loss': self.loss,
                    'weights': self.weights,
                    'biases': self.biases,
                    'config': {
                        'hyper_parameters': self.config.hyper_parameters.__dict__,
                        'data': self.config.data.serialize()
                    },
                    'predictions': {
                        'train': self.train_predictions,
                        'validate': self.validate_predictions,
                        'test': self.test_predictions
                    },
                    'accuracy': {
                        'train': self.train_accuracy,
                        'validate': self.validate_accuracy,
                        'test': self.test_accuracy
                    }
                }
            }
        else:
            return data

    def __accuracy_satisfies_minimum_requirements(self, accuracy_pct):
        """
        Compares an accuracy percentage to the minimum validation accuracy checkpoint threshold.

        :param accuracy_pct: A float corresponding to an accuracy measurement.
        :return: True if the accuracy percentage is greater than or equal to the minimum validation accuracy threshold.
        """
        return accuracy_pct >= self.MINIMUM_VALIDATION_ACCURACY_CHECKPOINT_THRESHOLD

    def __with_time(self, label, callback):
        start = datetime.now()
        print('')
        print('')
        print("===========> [{}] Started at {}".format(label, start.time()))
        print('')
        print('')

        callback()

        end = datetime.now()
        print('')
        print('')
        print("===========> [{}] Finished at {}".format(label, end.time()))
        print('')
        print("===========> [{}] Wall time: {}".format(label, end - start))
        print('')
        print("└[∵┌]   └[ ∵ ]┘   [┐∵]┘   └[ ∵ ]┘   └[∵┌]   └[ ∵ ]┘   [┐∵]┘   └[ ∵ ]┘   └[∵┌]")
        print('')
        print('')
        print('')

    def __generate_file_name(self):
        return '{}_{}_best_validation_{}S_{}LR_{}E_{}B'.format(
            self.__class__.__name__,
            self.uuid,
            "{:.002f}".format(self.config.data.split_size),
            "{:.004f}".format(self.config.hyper_parameters.start_learning_rate),
            self.config.hyper_parameters.epochs,
            self.config.hyper_parameters.batch_size)

    def __persist(self):
        if self.__accuracy_satisfies_minimum_requirements(self.best_validation_accuracy):
            TrainedDataSerializer.save_data(
                data=self.serialize(),
                pickle_file='{}_{}_trained_{}TA_{}VA_{}TestA_{}S_{}sLR_{}eLR_{}E_{}B.pickle'.format(
                    self.__class__.__name__,
                    self.uuid,
                    "{:.004f}".format(self.train_accuracy),
                    "{:.004f}".format(self.validate_accuracy),
                    "{:.004f}".format(self.test_accuracy),
                    "{:.004f}".format(self.config.data.split_size),
                    "{:.004f}".format(self.config.hyper_parameters.start_learning_rate),
                    "{:.004f}".format(self.config.hyper_parameters.end_learning_rate),
                    self.config.hyper_parameters.epochs,
                    self.config.hyper_parameters.batch_size),
                overwrite=True
            )

    # Helper functions

    def predict_cls(self, images, labels, cls_true, session, logits, features_placeholder, labels_placeholder):
        # Number of images.
        num_images = len(images)

        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        cls_pred = np.zeros(shape=num_images, dtype=np.int)

        # Now calculate the predicted classes for the batches.
        # We will just iterate through all the batches.
        # There might be a more clever and Pythonic way of doing this.

        # The starting index for the next batch is denoted i.
        i = 0

        while i < num_images:
            # The ending index for the next batch is denoted j.
            j = min(i + self.config.hyper_parameters.batch_size, num_images)

            # Create a feed-dict with the images and labels
            # between index i and j.
            feed_dict = {features_placeholder: images[i:j, :], labels_placeholder: labels[i:j, :]}

            # Calculate the predicted class using TensorFlow.
            y_pred_cls = tf.argmax(logits, dimension=1)
            y_true_cls = tf.argmax(labels_placeholder, 1)
            correct_prediction = tf.equal(y_pred_cls, y_true_cls)
            cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j

        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)

        return correct, cls_pred

    def __say_log(self, msg):
        print(msg)
        os.system('say "{}"'.format(msg))

    def __str__(self):
        result = []
        result.append(' ')
        for k, v in self.__dict__.items():
            result.append('{} - {}'.format(k, str(v)))
        result.append(' ')
        return '\n'.join(result)
