from datetime import datetime

from zimpy.datasets.german_traffic_signs import GermanTrafficSignDataset

import tensorflow as tf

data = GermanTrafficSignDataset()
data.configure(one_hot=True, train_validate_split_percentage=0)

# Parameters
learning_rate = 0.001
batch_size = 128
training_epochs = 1
dim_size = 3

n_classes = data.num_classes  # German Traffic Sign dataset total classes (0-42 signs)

layer_width = {
    'layer_1': 32,
    'layer_2': 64,
    'layer_3': 128,
    'fully_connected': 512
}

# Store layers weight & bias
weights = {
    'layer_1': tf.Variable(tf.truncated_normal(
        [5, 5, dim_size, layer_width['layer_1']])),
    'layer_2': tf.Variable(tf.truncated_normal(
        [5, 5, layer_width['layer_1'], layer_width['layer_2']])),
    'layer_3': tf.Variable(tf.truncated_normal(
        [5, 5, layer_width['layer_2'], layer_width['layer_3']])),
    'fully_connected': tf.Variable(tf.truncated_normal(
        [4 * 4 * 128, layer_width['fully_connected']])),
    'out': tf.Variable(tf.truncated_normal(
        [layer_width['fully_connected'], n_classes]))
}
biases = {
    'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
    'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
    'layer_3': tf.Variable(tf.zeros(layer_width['layer_3'])),
    'fully_connected': tf.Variable(tf.zeros(layer_width['fully_connected'])),
    'out': tf.Variable(tf.zeros(n_classes))
}


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.tanh(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')


# Create model
def conv_net(x, weights, biases):
    # Layer 1 - 32*32*3 to 16*16*32
    conv1 = conv2d(x, weights['layer_1'], biases['layer_1'])
    conv1 = maxpool2d(conv1)

    # Layer 2 - 16*16*32 to 8*8*64
    conv2 = conv2d(conv1, weights['layer_2'], biases['layer_2'])
    conv2 = maxpool2d(conv2)

    # Layer 3 - 8*8*64 to 4*4*128
    conv3 = conv2d(conv2, weights['layer_3'], biases['layer_3'])
    conv3 = maxpool2d(conv3)

    # Fully connected layer - 4*4*128 to 512
    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(
        conv3,
        [-1, weights['fully_connected'].get_shape().as_list()[0]])
    fc1 = tf.add(
        tf.matmul(fc1, weights['fully_connected']),
        biases['fully_connected'])
    fc1 = tf.nn.tanh(fc1)

    # Output Layer - class prediction - 512 to 43
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# tf Graph input
x = tf.placeholder("float", [None, 32, 32, 3])
y = tf.placeholder("float", [None, n_classes])

logits = conv_net(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) \
    .minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    start = datetime.now()
    label = 'cnn.py'

    print()
    print()
    print("===========> [{}] Started at {}".format(label, start.time()))
    print()
    print()

    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        e_start = datetime.now()

        total_batch = int(data.num_training / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y, start_i, end_i = data.next_orig_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        # Display logs per epoch step
        c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
        e_end = datetime.now()
        print("Epoch:", '%04d' % (epoch + 1), "wall time:", "{}".format(e_end - e_start))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(
        "Accuracy:",
        accuracy.eval({x: data.test_orig, y: data.test_labels}))

    end = datetime.now()
    print()
    print()
    print("===========> [{}] Finished at {}".format(label, end.time()))
    print()
    print("===========> [{}] Wall time: {}".format(label, end - start))
    print()
    print("└[∵┌]   └[ ∵ ]┘   [┐∵]┘   └[ ∵ ]┘   └[∵┌]   └[ ∵ ]┘   [┐∵]┘")
    print()
    print()
    print()
