'''
A linear regression learning algorithm example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import math
import time

start_time = time.time()
rng = numpy.random

# Parameters
# learning_rate = 0.1
training_epochs = 10000
display_step = 50
x_increment = 0.01
training_set_size_percent = 1.0
x_min = -2.0
x_max = 2.0
BATCH_SIZE = 100  # The number of training examples to use per training step.

# y = e^(-(1/2).x^2) . (4 sin^2 6 x + 3 cos^2 x . sin^2 4 x + 1)

number_of_x_values = math.floor(4.0/x_increment)
print "number_of_x_values: " + str(number_of_x_values)

set_x = []
set_y = []

x = x_min

for i in range(0, int(number_of_x_values)):
    set_x.append(float(x))
    y = math.exp(-0.5 * x * x) * ((4 * (math.sin(6.0 * x) * math.sin(6.0 * x))) + ((3.0 * math.cos(x) * math.cos(x)) * (math.sin(4.0 * x) * math.sin(4.0 * x))) + 1)
    set_y.append(y)
    x = x + x_increment
    if x > x_max:
        break

training_last_index = number_of_x_values * training_set_size_percent

set_x_train = set_x[0 : int(training_last_index)]
set_y_train = set_y[0 : int(training_last_index)]

print "training set size: " + str(len(set_x_train))

set_x_eval = set_x[int(training_last_index) : int(number_of_x_values)]
set_y_eval = set_y[int(training_last_index) : int(number_of_x_values)]

print "test set size: " + str(len(set_x_eval))



train_X = numpy.asarray(set_x_train)
train_Y = numpy.asarray(set_y_train)
n_samples = train_X.shape[0]
NUM_LABELS = n_samples    # The number of labels.

print "n_samples: " + str(n_samples)

# Training and evaluation set is ready

# Extract it into numpy matrices.
train_data = train_X
train_labels = train_Y
test_data = set_x_eval
test_labels = set_y_eval

# Get the shape of the training data.
print "train_data.shape: " + str(train_data.shape)
train_data = tf.reshape(train_data, [400, 1])
train_labels = tf.reshape(train_labels, [400, 1])
print "train_data.shape: " + str(train_data.get_shape())
train_size,num_features = train_data.get_shape()
print "train_size: " + str(train_size)
print "num_features: " + str(num_features)

# Get the number of epochs for training.
num_epochs = training_epochs

# This is where training samples and labels are fed to the graph.
# These placeholder nodes will be fed a batch of training data at each
# training step using the {feed_dict} argument to the Run() call below.
# x = tf.placeholder("float", shape=[None, num_features])
# y_ = tf.placeholder("float", shape=[None, NUM_LABELS])
x = tf.placeholder("float")
y_ = tf.placeholder("float")

# For the test data, hold the entire dataset in one constant node.
test_data_node = tf.constant(test_data)

# Define and initialize the network.

# These are the weights that inform how much each feature contributes to
# the classification.
W = tf.Variable(tf.zeros([num_features,NUM_LABELS]))
b = tf.Variable(tf.zeros([NUM_LABELS]))
print "x: " + str(type(x))
print "x.shape: " + str(x.get_shape())
print "W: " + str(type(W))
print "W: " + str(W.get_shape())
print "b: " + str(type(b))
print "b: " + str(b.get_shape())
y = tf.nn.softmax(tf.matmul(x,W) + b)

# Optimization.
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Evaluation.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Create a local session to run this computation.
with tf.Session() as s:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()

    print 'Initialized!'
    print
    print 'Training.'

    # Iterate and train.
    # print "num_epochs: " + str(type(num_epochs))
    # print "train_size: " + str(type(train_size))
    # print "BATCH_SIZE: " + str(type(BATCH_SIZE))
    print "Steps: " + str(num_epochs * train_size.value // BATCH_SIZE)
    for step in xrange(num_epochs * train_size.value // BATCH_SIZE):
        print step,

        offset = (step * BATCH_SIZE) % train_size.value
        batch_data = train_data[offset:(offset + BATCH_SIZE), :]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
        # print "batch_data: " + str(type(batch_data))
        # print "batch_labels: " + str(type(batch_labels))
        # print "batch_data.get_shape(): " + str(batch_data.get_shape())
        # print "batch_labels.get_shape(): " + str(batch_labels.get_shape())
        train_step.run(feed_dict={x: batch_data.eval(), y_: batch_labels.eval()})

        if offset >= train_size.value-BATCH_SIZE:
            print

    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    # Give very detailed output.

    print
    print 'Weight matrix.'
    print s.run(W)
    print
    print 'Bias vector.'
    print s.run(b)
    print
    # print "Applying model to first test instance."
    # first = [test_data[:1]]
    # first = tf.reshape(first, [1, 1])
    # # first = test_data
    # print "Point =", first
    # print "first: " + str(type(first))
    # print "first.shape: " + str(len(first))
    # print "W: " + str(type(W))
    # print "W: " + str(W.get_shape())
    # print "b: " + str(type(b))
    # print "b: " + str(b.get_shape())
    # print "Wx+b = ", s.run(tf.matmul(first,W)+b)
    # print "softmax(Wx+b) = ", s.run(tf.nn.softmax(tf.matmul(first,W)+b))
    # print

    # print "Accuracy:", accuracy.eval(feed_dict={x: test_data, y_: test_labels})

    plt.rc('font', family='serif')
    plt.xlabel(r'x')
    # y = e^(-(1/2).x^2) . (4 sin^2 6 x + 3 cos^2 x . sin^2 4 x + 1)
    plt.ylabel(r'$e^{-\frac{1}{2} x^2} \left(4 \sin^2 6 x + 3 \cos^2 x \sin^2 4 x + 1\right)$')
    plt.ylim(-2, 10)
    print "train_X: " + str(train_X.shape)
    print "train_Y: " + str(train_Y.shape)
    # print "train_X: \n" + str(train_X)
    # print "train_Y: \n" + str(train_Y)
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    print "train_data: " + str(train_data.get_shape())
    print "s.run(W) * train_X + s.run(b): " + str((s.run(W) * train_X + s.run(b)).shape)
    print "train_data.eval(): " + str(train_data.eval().shape)
    print "numpy.transpose(s.run(W) * train_X + s.run(b)): " \
          + str(numpy.transpose(s.run(W) * train_X + s.run(b)).shape)
    # print "train_data.eval(): \n" + str(train_data.eval())
    # print "s.run(W) * train_X + s.run(b): \n" + str(numpy.transpose(s.run(W) * train_X + s.run(b)))
    plt.plot(train_data.eval(), numpy.transpose(s.run(W) * train_X + s.run(b)), label='Fitted line')
    plt.legend()
    text = "Training epochs: " + str(training_epochs) \
           + ", method: Gradient Descent, Batch size: " + str(BATCH_SIZE) +",\n Elapsed time: " + str(hours) + " hours, " \
           + str(minutes) + " minutes, " + str(seconds) + " seconds."
    plt.figtext(0, 0, text, color='black')
    plt.savefig("gd-logir-epoch-" + str(training_epochs) + "-batch-size-" + str(BATCH_SIZE) +".pdf")

