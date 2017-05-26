# Image denoising using MRF model

# PIL can handle the BMP inputs but not the PNG inputs.

from PIL import Image
import numpy
numpy.set_printoptions(threshold=numpy.nan)
from pylab import *
from sympy import *
import sys
import os
import logging
from logging.handlers import RotatingFileHandler
import time
from sympy.parsing.sympy_parser import *
import math

# Setting up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a log file handler
logFile = 'logs/mrf_' + time.strftime("%d-%m-%Y") + '.log'

# handler = logging.FileHandler(logFile)
handler = RotatingFileHandler(logFile, mode='a', maxBytes=100*1024*1024, backupCount=100, encoding=None, delay=0)
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

variables = []

def main():
        # Configurations
        binary_threshold = 100
        dw_range = 4

        experiment_start = time.time()
        binary_image_creation_start_time = time.time()

	# Read in image
        name = 'lena512-32'
        logger.info( "Reading the image...")

	im=Image.open(name + '.bmp')
	im=numpy.array(im)
        logger.info("Shape of the original image: " + str(im.shape))
        logger.info("Converting into binary...")
	im=where (im>100,1,0) #convert to binary image

        logger.info("Saving the binary image...")
        (Image.fromarray(np.uint8(cm.gist_earth(im) * 255))).save(name + '-binary.bmp')

        binary_image_creation_end_time = time.time()
        binary_image_creation_elapsed_time = binary_image_creation_end_time - binary_image_creation_start_time
        binary_image_creation_hours, binary_image_creation_rem = divmod(binary_image_creation_elapsed_time, 3600)
        binary_image_creation_minutes, binary_image_creation_seconds = divmod(binary_image_creation_rem, 60)
        binary_image_creation_time_log_string = "Binary image creation time: " + "{:0>2}:{:0>2}:{:05.2f}".format(int(binary_image_creation_hours), int(binary_image_creation_minutes), binary_image_creation_seconds)
        logger.info(binary_image_creation_time_log_string)


	(M,N)=im.shape

        add_random_noise_start_time = time.time()
	# Add noise
        logger.info("Adding random noise to the image...")
	noisy=im.copy()
	noise=numpy.random.rand(M,N)
	ind=where(noise<0.2)
	noisy[ind]=1-noisy[ind]
        logger.info("Saving the noisy image...")
        (Image.fromarray(np.uint8(cm.gist_earth(noisy) * 255))).save(name + '-noisy.bmp')

        add_random_noise_end_time = time.time()
        add_random_noise_elapsed_time = add_random_noise_end_time - add_random_noise_start_time
        add_random_noise_hours, add_random_noise_rem = divmod(add_random_noise_elapsed_time, 3600)
        add_random_noise_minutes, add_random_noise_seconds = divmod(add_random_noise_rem, 60)
        add_random_noise_time_log_string = "Add random noise time: " + "{:0>2}:{:0>2}:{:05.2f}".format(int(add_random_noise_hours), int(add_random_noise_minutes), add_random_noise_seconds)
        logger.info(add_random_noise_time_log_string)

        expression_creation_start_time = time.time()
        logger.info("Restoring the image...")
	out=MRF_denoise_sym(noisy)

        logger.info("The expression...")
        logger.info(str(out))    
  
        logger.info("Opening the file to write the expression...")
        target = open(name + '-denoise-expression.txt', 'w')

        # Resettingthe recursion limit from 1000 to 10000
        sys.setrecursionlimit(10000)

        logger.info("Writing the expression to "  + name + "-denoise-expression.txt")  
        out_str = str(out)
        logger.info("Length of the expression: " + str(len(out_str)))  
        target.write(out_str)
        target.flush()
        os.fsync(target.fileno())
        # Now write the Matlab format
        logger.info("Converting into Mathematica format...")  
        out_str_mat = out_str.replace("*", " ")
        out_str_mat = out_str_mat.replace("_", "")
        target.write("\n\n" + out_str_mat)
        target.flush()
        os.fsync(target.fileno())
        logger.info("Writing finished")
        logger.info("Closing the file")    
        target.close()
        logger.info("Closed")    


        logger.info("Opening the file to write the symplified expression...")
        target_simpl = open(name + '-symplified-denoise-expression.txt', 'w')

        logger.info("Writing the symplified expression to "  + name + "-denoise-expression.txt")  
        out_sym = expand(out)
        logger.info("The simplified expression...")
        logger.info(str(out_sym))    

        out_sym_str = str(out_sym)
        logger.info("Length of the expression: " + str(len(out_sym_str)))  
        target_simpl.write(out_sym_str)
        target_simpl.flush()
        os.fsync(target_simpl.fileno())
        # Now write the Matlab format
        logger.info("Converting into Mathematica format...")  
        out_sym_str_mat = out_sym_str.replace("*", " ")
        out_sym_str_mat = out_sym_str_mat.replace("_", "")
        target_simpl.write("\n\n" + out_sym_str_mat)
        target_simpl.flush()
        os.fsync(target_simpl.fileno())

        logger.info("Writing finished")  
        logger.info("Closing the file")    
        target_simpl.close()
        logger.info("Closed")    

        logger.info("Degree list of the expression...")  
        deg_list = degree_list(out)
        logger.info(str(deg_list))
        logger.info("Size of degree list: " + str(len(deg_list)))
        logger.info("Number of variables: " + str(len(variables)))        

        for var in variables:
           var_str = str(var)
           logger.info("Occurence of " + var_str + " is " + str(out_sym_str.count(var_str)))

        logger.info("Creating the dictionary of coefficients...")
        term_dict = out_sym.as_coefficients_dict()
        logger.info("Size of the dictionary:")
        logger.info(len(term_dict))
        logger.info("Removing the constant...")
        term_dict.pop(1, None)
        logger.info("Size of the dictionary:")
        logger.info(len(term_dict))


        if len(term_dict) == len(variables):
           logger.info("The number of term is equal to the number of variables.")
        else:
           logger.info("The number of term is not equal to the number of variables.")

        logger.info("Coefficients as dict:")
        logger.info(str(term_dict))

        logger.info("Maximum coefficient:")
        maximum = max(term_dict, key=term_dict.get)  
        logger.info(str((maximum, term_dict[maximum])))

        logger.info("Minimum coefficient:")
        minimum = min(term_dict, key=term_dict.get)  
        logger.info(str((minimum, term_dict[minimum])))

        norm_factor = math.ceil((term_dict[maximum] - term_dict[minimum]) / 4)
        logger.info("Normalizing with factor:" + str(norm_factor))

        logger.info("Normalized expression with fractions:")
        out_sym_norm = out_sym / ((term_dict[maximum] - term_dict[minimum]) / 4)
        logger.info(str(out_sym_norm))


        logger.info("Normalized expression with decimal points:")
        out_sym_norm = out_sym / norm_factor
        logger.info(str(out_sym_norm))

        logger.info("Creating the dictionary of coefficients...")
        term_dict = out_sym_norm.as_coefficients_dict()
        logger.info("Size of the dictionary:")
        logger.info(len(term_dict))
        logger.info("Removing the constant...")
        term_dict.pop(1, None)
        logger.info("Size of the dictionary:")
        logger.info(len(term_dict))


        logger.info("Maximum coefficient:")
        maximum = max(term_dict, key=term_dict.get)  
        logger.info(str((maximum, term_dict[maximum])))

        logger.info("Minimum coefficient:")
        minimum = min(term_dict, key=term_dict.get)  
        logger.info(str((minimum, term_dict[minimum])))


        less_than_one_third = 0
        less_than_two_to_seventh = 0

        logger.info("Free symbols: ")
        logger.info(str(out_sym_norm.free_symbols))

        logger.info("Creating diagonal vectors...")
        logger.info("List of variables...")
        logger.info(str(variables))
        diagonal = []
        for var in variables:
           term = term_dict[var]
           if term < (1/3):
              less_than_one_third = less_than_one_third + 1
           else:
              logger.info(str(term) + " is greater than 1/3")
           if term < (1/(2**7)):
              less_than_two_to_seventh = less_than_two_to_seventh + 1
           else:
              logger.info(str(term) + " is greater than 1/2^7")


           diagonal.append(term_dict[var])
        logger.info(str(diagonal))
        logger.info("Maximum: " + str(max(diagonal)))
        logger.info("Minimum: " + str(min(diagonal)))

        logger.info(str(less_than_one_third) + " out of " + str(len(variables)) + " coefficients are less than 1/3")
        logger.info(str(less_than_two_to_seventh) + " out of " + str(len(variables)) + " coefficients are less than 1/2^7")

        logger.info("Creating off-diagonal matrix")
        off_diagonal = []
        for var1 in variables:
           for var2 in variables:
              if var1 != var2:
                 if  (var1 * var2) in term_dict:
                    off_diagonal[var1 * var2] = term_dict[var1 * var2]
        logger.info(str(off_diagonal))

        expression_creation_end_time = time.time()
        expression_creation_elapsed_time = expression_creation_end_time - expression_creation_start_time
        expression_creation_hours, expression_creation_rem = divmod(expression_creation_elapsed_time, 3600)
        expression_creation_minutes, expression_creation_seconds = divmod(expression_creation_rem, 60)
        expression_creation_time_log_string = "Expression creation time: " + "{:0>2}:{:0>2}:{:05.2f}".format(int(expression_creation_hours), int(expression_creation_minutes), expression_creation_seconds)
        logger.info(expression_creation_time_log_string)


        experiment_end_time = time.time()
        experiment_elapsed_time = experiment_end_time - experiment_start_time
        experiment_hours, experiment_rem = divmod(experiment_elapsed_time, 3600)
        experiment_minutes, experiment_seconds = divmod(experiment_rem, 60)
        experiment_time_log_string = "Expression creation time: " + "{:0>2}:{:0>2}:{:05.2f}".format(int(experiment_hours), int(experiment_minutes), experiment_seconds)
        logger.info(experiment_time_log_string)


def MRF_denoise_sym(noisy):
        print "MRF_denoise_sym()..."
	# Start MRF	
	(M,N)=noisy.shape
	print "Shape of the image: " + str((M, N))
	y_old=noisy
	# y=zeros((M,N))
        y = MatrixSymbol('Y', M, N)

        final_cost_expression = 0
        print "For loops starting..."
	for i in range(M):
                print "Current row: " + str(i) 
                logger.info("Current row: " + str(i) )
                # print final_cost_expression
                logger.info(str(final_cost_expression))
		for j in range(N):
                        logger.info("Current column: " + str(j) )
			index=neighbor(i,j,M,N)
                        logger.info("Neighbors: " + str(index) )
			x = symbols('x_' + str(i) + "_" + str(j))
                        variables.append(x)
			cost_expression_for_pixel = cost_sym(x, noisy[i,j], y_old, index)
                        logger.info("Cost expression for pixel: " + str(cost_expression_for_pixel) )
                        final_cost_expression = final_cost_expression + cost_expression_for_pixel


        # print final_cost_expression
	return ((-1) * final_cost_expression)



# This function is a symbolic version of delta function
# for this project.
def delta_sym(a, b):
        delta_expr = ((a * b) + ((1 - a) * (1 - b)))
        logger.info("Delta function: " + str(delta_expr))
        return delta_expr

def neighbor(i,j,M,N):
	#find correct neighbors
	if (i==0 and j==0):
		neighbor=[(0,1), (1,0)]
	elif i==0 and j==N-1:
		neighbor=[(0,N-2), (1,N-1)]
	elif i==M-1 and j==0:
		neighbor=[(M-1,1), (M-2,0)]
	elif i==M-1 and j==N-1:
		neighbor=[(M-1,N-2), (M-2,N-1)]
	elif i==0:
		neighbor=[(0,j-1), (0,j+1), (1,j)]
	elif i==M-1:
		neighbor=[(M-1,j-1), (M-1,j+1), (M-2,j)]
	elif j==0:
		neighbor=[(i-1,0), (i+1,0), (i,1)]
	elif j==N-1:
		neighbor=[(i-1,N-1), (i+1,N-1), (i,N-2)]
	else:
		neighbor=[(i-1,j), (i+1,j), (i,j-1), (i,j+1),\
				  (i-1,j-1), (i-1,j+1), (i+1,j-1), (i+1,j+1)]
	# print "The neighbers of (" + str(i) + ", " + str(j) + ")-th pixel are: " + str(neighbor)
	return neighbor

def neighbors_to_variables(var, neighbors):
        expression = "0 "
        for tuple in neighbors:
           expression = expression + " + "  + var + "_" + str(tuple[0]) + "_" + str(tuple[1]) + " "
        return expression


def cost_sym(y, x, y_old, index):
        alpha = 1
        beta = 10

        term1 = alpha * delta_sym(y,x)
        logger.info("Term 1: " + str(term1))
        term2 = beta * sum(delta_sym(y,y_old[i]) for i in index)
        logger.info("Term 2: " + str(term2))

        cost_sym = term1 + term2
        return cost_sym


if __name__=="__main__":
	main()
