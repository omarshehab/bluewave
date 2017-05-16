# This program adds salt and pepper noise to an image.

from PIL import Image
import scipy
from scipy.misc import imread
import numpy as np
import os
import cv2
from skimage import img_as_float, io
from skimage.util import random_noise

def noisy():
   image_input_file = "g/cifarjpg/1"
   image_output_file = "g/cifarjpgsp/1"

   print "Reading image as an array: "
   # img = imread("a2.png")
   image = io.imread(image_input_file + ".jpg")
   print "Image as an array: " + str(image)

   noisy = random_noise(image, mode='s&p')
   
   io.imsave(image_output_file + ".jpg", noisy)
