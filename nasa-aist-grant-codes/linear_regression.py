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
learning_rate = 0.0001
training_epochs = 1000
display_step = 50
x_increment = 0.01
training_set_size_percent = 1.0
x_min = -2.0
x_max = 2.0

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

# Training Data
# train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
#                          7.042,10.791,5.313,7.997,5.654,9.27,3.1])
# train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
#                          2.827,3.465,1.65,2.904,2.42,2.94,1.3])

train_X = numpy.asarray(set_x_train)
train_Y = numpy.asarray(set_y_train)
n_samples = train_X.shape[0]

print "n_samples: " + str(n_samples)

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.mul(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2.0*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b)

    print "Optimization Finished!"
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'

    #Graphic display
    # plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel(r'x')
    # y = e^(-(1/2).x^2) . (4 sin^2 6 x + 3 cos^2 x . sin^2 4 x + 1)
    plt.ylabel(r'$e^{-\frac{1}{2} x^2} \left(4 \sin^2 6 x + 3 \cos^2 x \sin^2 4 x + 1\right)$')
    plt.ylim(-2, 10)
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    text = "Learning rate: " + str(learning_rate) + ", training epochs: " + str(training_epochs) \
           + ", method: Gradient Descent,\n Elapsed time: " + str(hours) + " hours, " \
           + str(minutes) + " minutes, " + str(seconds) + " seconds."
    plt.figtext(0, 0, text, color='black')
    plt.savefig("gd-lr-" + str(learning_rate) +".pdf")
    # plt.show()

