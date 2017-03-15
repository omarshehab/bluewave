from cycler import cycler
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import itertools
from matplotlib.font_manager import FontProperties
from itertools import cycle
import csv

training_epochs = 4001
hiddenDim = 10
NUM_COLORS = 7

# list_of_rmses = []
# list_of_neurons = []
# list_of_epochs = []


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Can't make tensorflow consume ordinary lists unless they're parsed to ndarray
def toNd(lst):
    lgt = len(lst)
    x = np.zeros((1, lgt), dtype='float32')
    for i in range(0, lgt):
        x[0,i] = lst[i]
    return x

resultFile = open('result.csv', 'a')
resultCSV = csv.writer(resultFile)

lines = ["-","--","-.",":"]
linecycler = cycle(lines)
xTest = np.linspace(-2.0, 2.0, 1001)
fig = plt.figure()
plt.rcParams.update({'font.size': 8})
plt.xlabel('xlabel', fontsize=8)
plt.ylabel('ylabel', fontsize=8)
plt.rc('font', family='serif')
# plt.xlabel(r'x')
# plt.xlabel("Neurons")
plt.xlabel("Neurons")
# plt.ylabel(r'$e^{-\frac{1}{2} x^2} \left(4 \sin^2 6 x + 3 \cos^2 x \sin^2 4 x + 1\right)$')
plt.ylabel("RMSE")
# plt.ylim(0, 2)
cm = plt.get_cmap('gist_rainbow')
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y'])))
ax = plt.subplot(111)
ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
# plt.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])


# y = e^(-(1/2).x^2) . (4 sin^2 6 x + 3 cos^2 x . sin^2 4 x + 1)
# plt.plot(xTest,map(lambda x: math.exp(-0.5 * x * x) * ((4 * (math.sin(6.0 * x) * math.sin(6.0 * x))) + ((3.0 * math.cos(x) * math.cos(x)) * (math.sin(4.0 * x) * math.sin(4.0 * x))) + 1), xTest), label='Original data')

# for training_epochs in range(10000, 50001, 20000):
# for training_epochs in range(60000, 100001, 20000):
# for training_epochs in range(40000, 100001, 1000):
# for training_epochs in range(95000, 100001, 1000):
list_of_rmses = []
list_of_neurons = []
list_of_epochs = []
# list_of_neurons = []

for hiddenDim in range(10, 51, 1):
    # list_of_rmses = []
    # list_of_epochs = []
    # list_of_neurons = []
    for training_epochs in range(100000, 100001, 5000):
        for optimizer_count in range(0, 1, 1):
            start_time = time.time()
            print "training_epochs: " + str(training_epochs)
            print "hiddenDim: " + str(hiddenDim)
            print "optimizer_count: " + str(optimizer_count)
            xBasic = np.linspace(-2.0, 2.0, 401)
            xTrain = toNd(xBasic)
            yTrain = toNd(map(lambda x: math.exp(-0.5 * x * x) * ((4 * (math.sin(6.0 * x) * math.sin(6.0 * x))) + ((3.0 * math.cos(x) * math.cos(x)) * (math.sin(4.0 * x) * math.sin(4.0 * x))) + 1), xBasic))

            x = tf.placeholder("float", [1,None])


            b = bias_variable([hiddenDim,1])
            W = weight_variable([hiddenDim, 1])

            b2 = bias_variable([1])
            W2 = weight_variable([1, hiddenDim])

            hidden = tf.nn.sigmoid(tf.matmul(W, x) + b)
            y = tf.matmul(W2, hidden) + b2

            # Minimize the root mean squared errors.
            # tf.sqrt(tf.reduce_mean(tf.square(tf.sub(targets, outputs))))
            loss = tf.sqrt(tf.reduce_mean(tf.square(y - yTrain)))
            # optimizer = tf.train.GradientDescentOptimizer(0.5)
            optimizer = tf.train.AdadeltaOptimizer(0.5)
            optimizer_label = ""

            # if optimizer_count == 0:
            #     continue
            #     # optimizer = tf.train.GradientDescentOptimizer(0.5)
            #     # optimizer_label = "Gradient Descent"
            # if optimizer_count == 1:
            #     continue
            #     # optimizer = tf.train.AdadeltaOptimizer(0.5)
            #     # optimizer_label = "Adadelta"
            if optimizer_count == 0:
                optimizer = tf.train.AdagradOptimizer(0.5)
                optimizer_label = "Adagrad"
            # if optimizer_count == 3:
            #     continue
            #     # optimizer = tf.train.MomentumOptimizer(0.5, 0.9)
            #     # optimizer_label = "Momentum"
            # if optimizer_count == 1:
            #     optimizer = tf.train.AdamOptimizer(0.5)
            #     optimizer_label = "Adam"
            # if optimizer_count == 5:
            #     continue
            #     # optimizer = tf.train.FtrlOptimizer(0.5)
            #     # optimizer_label = "FTRL"
            # if optimizer_count == 6:
            #     continue
            #     # optimizer = tf.train.RMSPropOptimizer(0.5)
            #     # optimizer_label = "RMSProp"


            train = optimizer.minimize(loss)

            # For initializing the variables.
            init = tf.initialize_all_variables()

            # Launch the graph
            sess = tf.Session()
            sess.run(init)

            for step in xrange(0, training_epochs):
                train.run({x: xTrain}, sess)
                if step % 500 == 0:
                    accuracy = loss.eval({x: xTrain}, sess)


            list_of_rmses.append(accuracy)
            # list_of_neurons.append(hiddenDim)
            # list_of_epochs.append(training_epochs)
            list_of_neurons.append(hiddenDim)
            yTest = y.eval({x:toNd(xTest)}, sess)
            test_loss = tf.sqrt(tf.reduce_mean(tf.square(y - yTest)))
            # abs_cost = loss - test_loss
            # ccuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            # accuracy = test_loss.eval(session=sess, feed_dict={x: xTest})

            elapsed_time = time.time() - start_time
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)

            row = [[training_epochs, hiddenDim, accuracy, hours, minutes, seconds]]
            resultCSV.writerows(row)
print "list_of_epochs: " + str(len(list_of_neurons))
print "list_of_rmses: " + str(len(list_of_rmses))
            # plt.plot(xTest,yTest.transpose().tolist())
ax.plot(list_of_neurons,list_of_rmses, next(linecycler),
                label=optimizer_label)


        # text = "Epochs: " + str(training_epochs) \
        #            + ", Hidden layer neurons: " + str(hiddenDim) + ",\n Elapsed time: " + str(hours) + " hours, " \
        #            + str(minutes) + " minutes, " + str(seconds) + " seconds."
        # plt.figtext(0, 0, text, color='black')




# plt.plot(list_of_neurons, list_of_rmses)

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1),
          ncol=2, fancybox=False, shadow=False)
plt.legend()
# plt.savefig("gd-nlr-epochs-" + str(training_epochs) + "-hidden-layer-neurons-" + str(hiddenDim) + ".pdf")
plt.savefig("adagrad-optimizers-neuron-10-50-ep-100k.pdf")
resultFile.close()
        #plt.show()