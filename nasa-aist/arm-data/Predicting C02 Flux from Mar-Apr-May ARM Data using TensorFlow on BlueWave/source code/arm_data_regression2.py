# https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/input_fn/boston.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import itertools
import pandas as pd
import tensorflow as tf
import logging
from logging.handlers import RotatingFileHandler
import time
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import plotly.plotly as py
from sklearn import preprocessing

# Setting up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a log file handler
logFile = '../logs/arm_data_regression2_' + time.strftime("%d-%m-%Y") + '.log'

# handler = logging.FileHandler(logFile)
handler = RotatingFileHandler(logFile, mode='a', maxBytes=100*1024*1024, backupCount=100, encoding=None, delay=0)
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

# Set TF log level
tf.logging.set_verbosity(tf.logging.DEBUG)

COLUMNS = ["Date", "CO2Flux", "CO2", "H2O", "Temperature", "Pressure", "WindSpeed","HorizontalWindDirection", "RotationToZeroWTheta", "RotationToZeroVPhi", "SensibleHeatFlux", "LatentHeatFlux", "FrictionVelocity"]

FEATURES =  ["Date", "CO2", "H2O", "Temperature", "Pressure", "WindSpeed","HorizontalWindDirection", "RotationToZeroWTheta", "RotationToZeroVPhi", "SensibleHeatFlux", "LatentHeatFlux", "FrictionVelocity"]

LABEL = "CO2Flux"


def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels


def main():
  # Start the timer
  experiment_start = time.time()
  logger.info("\n\n\n\nExperiment started --------------------------------------------------------------------------------")
  
  # Load datasets
  training_set = pd.read_csv("../curated data set/ARM4mDec2002Jul2015OklahomaV2_mar_apr_may_date_time_normalized_16000_training_data.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)
  logger.info("Training set loaded into a numpy array...")
  test_set = pd.read_csv("../curated data set/ARM4mDec2002Jul2015OklahomaV2_mar_apr_may_date_time_normalized_8000_test_data.csv", skipinitialspace=True,
                         skiprows=1, names=COLUMNS)
  logger.info("Test data loaded into a numpy array...")

  # 
  prediction_set = pd.read_csv("../curated data set/ARM4mDec2002Jul2015OklahomaV2_mar_apr_may_date_time_normalized_8000_validate_data.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
  logger.info("Validation data loaded into a numpy array...")

  # Feature cols
  feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]
  logger.info("Feature columns created...")
 
  # Normalizing features
  # Scale data (training set) to 0 mean and unit standard deviation.
  # logger.info("Normalizing feature columns...")
  # logger.info("Type of training set: " + str(type(training_set)))
  # logger.info("Training set before normalization:")
  # logger.info(str(training_set))
  # training_set[FEATURES] = preprocessing.normalize(training_set[FEATURES])
  # test_set[FEATURES] = preprocessing.normalize(test_set[FEATURES])
  # prediction_set[FEATURES] = preprocessing.normalize(prediction_set[FEATURES])
  # observations = preprocessing.normalize(observations) 
  # logger.info("Training set after normalization:")
  # logger.info(str(training_set))
  # logger.info("Normalizing complete.")
  # logger.info("Type of training set: " + str(type(training_set)))

  # Build 2 layer fully connected DNN with 10, 10 units respectively.
  number_of_hidden_layers = 10
  number_of_neurons_per_layer = 20
  hidden_unit_array = list(itertools.repeat(number_of_neurons_per_layer, number_of_hidden_layers))  # 10 copies of 20 neurons
  learning_rate_value = 0.00015
  steps_value = 1000
  logger.info("Creating a DNNRegressor with following neural network")
  logger.info("Hidden units: " + str(hidden_unit_array))
  logger.info("Learning rate: " + str(learning_rate_value))
  logger.info("Steps: " + str(steps_value))
  # This is only for AdagradDAOptimizer
  # global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
  # sess = tf.Session()
  # tf.train.global_step(sess, global_step_tensor)
  training_start_time = time.time()
  regressor = tf.contrib.learn.DNNRegressor(
      feature_columns=feature_cols, hidden_units=hidden_unit_array, optimizer=tf.train.GradientDescentOptimizer(
      learning_rate=learning_rate_value), enable_centered_bias = True)

  # Fit
  logger.info("Fitting under way")
  regressor.fit(input_fn=lambda: input_fn(training_set), steps=steps_value)
  
  training_end_time = time.time()
  training_elapsed_time = training_end_time - training_start_time
  training_hours, training_rem = divmod(training_elapsed_time, 3600)
  training_minutes, training_seconds = divmod(training_rem, 60)
  training_time_log_string = "Training time: " + "{:0>2}:{:0>2}:{:05.2f}".format(int(training_hours), int(training_minutes), training_seconds)
  logger.info(training_time_log_string)


  # Score accuracy
  logger.info("Computing score accuracy...")
  ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
  logger.info("Output of DNNRegressor evaluate(): \n" + str(ev))
  loss_score = ev["loss"]
  print("Loss: {0:f}".format(loss_score))

  # Print out predictions
  logger.info("Validating...")
  validation_start_time = time.time()
  y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
  logger.info("Type of validation output: " + str(type(y)))
  logger.info("Iterator for predictions created... converting it into a list...")
  # .predict() returns an iterator; convert to a list and print predictions
  predictions = list(itertools.islice(y, 7995))
  logger.info("Conversion completed.")
  logger.info(str(predictions))
  # print("Predictions: {}".format(str(predictions)))
  
  validation_end_time = time.time()
  validation_elapsed_time = validation_end_time - validation_start_time
  validation_hours, validation_rem = divmod(validation_elapsed_time, 3600)
  validation_minutes, validation_seconds = divmod(validation_rem, 60)
  validation_time_log_string = "Validation time: " + "{:0>2}:{:0>2}:{:05.2f}".format(int(validation_hours), int(validation_minutes), validation_seconds)
  logger.info(validation_time_log_string)

  logger.info("Computing RMSE...")
  # Compute the RMSE from predictins and observations
  observations_list = prediction_set[["CO2Flux"]].values.tolist()
  rmse = sqrt(mean_squared_error(observations_list, predictions))
  logger.info("RMSE: " + str(rmse))

  logger.info("Creating the plot...")
  fig = plt.figure()
  plt.subplots_adjust(top=0.85)
  plt.scatter(observations_list, predictions, color='b', edgecolors = 'k', marker='.')
  fig.canvas.mpl_connect('draw_event', on_draw)
  logger.info("Scatter method returned successfully")
  plt.xlabel('Observations')
  plt.ylabel('Predictions')
  logger.info("Axis labels created")
  title_text = "GradientDescentOptimizer, number of hidden layers " + str(len(hidden_unit_array)) + ", steps " + str(steps_value) + ", learning rate " + str(learning_rate_value) + ", rmse " + str(rmse) + ", " + training_time_log_string + ", " + validation_time_log_string
  plt.title(title_text)
  plt.ylim([-40, 30])
  plt.xlim([-40, 30])
  plt.grid()
  logger.info("Title set")
  logger.info("Saving the plot...")
  plt.savefig("../plots/" + "arm_data_16000_" + title_text + "_" + time.strftime("%d-%m-%Y")  + ".png", dpi=1200)
  
  experiment_end = time.time()
  elapsed = experiment_end - experiment_start
  hours, rem = divmod(elapsed, 3600)
  minutes, seconds = divmod(rem, 60)
  log_string = "Time spent: " + "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
  logger.info( log_string)
  log_string = "------|||||| END OF EXPERIMENT ||||||------"
  logger.info( log_string)

if __name__ == "__main__":
  tf.app.run()

def on_draw(event):
    """Auto-wraps all text objects in a figure at draw-time"""
    import matplotlib as mpl
    fig = event.canvas.figure

    # Cycle through all artists in all the axes in the figure
    for ax in fig.axes:
        for artist in ax.get_children():
            # If it's a text artist, wrap it...
            if isinstance(artist, mpl.text.Text):
                autowrap_text(artist, event.renderer)

    # Temporarily disconnect any callbacks to the draw event...
    # (To avoid recursion)
    func_handles = fig.canvas.callbacks.callbacks[event.name]
    fig.canvas.callbacks.callbacks[event.name] = {}
    # Re-draw the figure..
    fig.canvas.draw()
    # Reset the draw event callbacks
    fig.canvas.callbacks.callbacks[event.name] = func_handles

def autowrap_text(textobj, renderer):
    """Wraps the given matplotlib text object so that it exceed the boundaries
    of the axis it is plotted in."""
    import textwrap
    # Get the starting position of the text in pixels...
    x0, y0 = textobj.get_transform().transform(textobj.get_position())
    # Get the extents of the current axis in pixels...
    clip = textobj.get_axes().get_window_extent()
    # Set the text to rotate about the left edge (doesn't make sense otherwise)
    textobj.set_rotation_mode('anchor')

    # Get the amount of space in the direction of rotation to the left and 
    # right of x0, y0 (left and right are relative to the rotation, as well)
    rotation = textobj.get_rotation()
    right_space = min_dist_inside((x0, y0), rotation, clip)
    left_space = min_dist_inside((x0, y0), rotation - 180, clip)

    # Use either the left or right distance depending on the horiz alignment.
    alignment = textobj.get_horizontalalignment()
    if alignment is 'left':
        new_width = right_space 
    elif alignment is 'right':
        new_width = left_space
    else:
        new_width = 2 * min(left_space, right_space)

    # Estimate the width of the new size in characters...
    aspect_ratio = 0.5 # This varies with the font!! 
    fontsize = textobj.get_size()
    pixels_per_char = aspect_ratio * renderer.points_to_pixels(fontsize)

    # If wrap_width is < 1, just make it 1 character
    wrap_width = max(1, new_width // pixels_per_char)
    try:
        wrapped_text = textwrap.fill(textobj.get_text(), wrap_width)
    except TypeError:
        # This appears to be a single word
        wrapped_text = textobj.get_text()
    textobj.set_text(wrapped_text)

def min_dist_inside(point, rotation, box):
    """Gets the space in a given direction from "point" to the boundaries of
    "box" (where box is an object with x0, y0, x1, & y1 attributes, point is a
    tuple of x,y, and rotation is the angle in degrees)"""
    from math import sin, cos, radians
    x0, y0 = point
    rotation = radians(rotation)
    distances = []
    threshold = 0.0001 
    if cos(rotation) > threshold: 
        # Intersects the right axis
        distances.append((box.x1 - x0) / cos(rotation))
    if cos(rotation) < -threshold: 
        # Intersects the left axis
        distances.append((box.x0 - x0) / cos(rotation))
    if sin(rotation) > threshold: 
        # Intersects the top axis
        distances.append((box.y1 - y0) / sin(rotation))
    if sin(rotation) < -threshold: 
        # Intersects the bottom axis
        distances.append((box.y0 - y0) / sin(rotation))
    return min(distances)
