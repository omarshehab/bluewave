# Image denoising using MRF model

# PIL can handle the BMP inputs but not the PNG inputs.

from PIL import Image
import numpy
numpy.set_printoptions(threshold=numpy.nan)
from pylab import *
from sympy import *

def main():

	# Read in image
        name = 'lena512-32'
        print "Reading the image..."
	im=Image.open(name + '.bmp')
	im=numpy.array(im)
        print "Shape: " + str(im.shape)
        print "Converting into binary..."
	im=where (im>100,1,0) #convert to binary image
        print "Shape: " + str(im.shape)
        (Image.fromarray(np.uint8(cm.gist_earth(im) * 255))).save(name + '-binary.bmp')
	(M,N)=im.shape

	# Add noise
        print "Adding noise..."
	noisy=im.copy()
	noise=numpy.random.rand(M,N)
	ind=where(noise<0.2)
	noisy[ind]=1-noisy[ind]
        print "Saving noisy image..."
        (Image.fromarray(np.uint8(cm.gist_earth(noisy) * 255))).save(name + '-noisy.bmp')

	# gray()
	# title('Noisy Image')
	# imshow(noisy)

	out=MRF_denoise(noisy)
        im = Image.fromarray(np.uint8(cm.gist_earth(out)*255))
     
        im.save(name + '-denoised.bmp')
	
	# figure()		
	# gray()
	# title('Denoised Image')
	# imshow(out)
	# show()

def MRF_denoise(noisy):
        print "MRF_denoise()..."
	# Start MRF	
	(M,N)=noisy.shape
	print "Shape of the image: " + str((M, N))
	y_old=noisy
	# y=zeros((M,N))
        y = MatrixSymbol('Y', M, N)

        print "While loop starting..."
	while(SNR(y_old,y)>0.01):
		print SNR(y_old,y)
		for i in range(M):
			for j in range(N):
                                print "Current pixel: " + str(i) + ", " + str(j)
				index=neighbor(i,j,M,N)
				
				a=cost(1,noisy[i,j],y_old,index)
				b=cost(0,noisy[i,j],y_old,index)

				if a>b:
					y[i,j]=1
				else:
					y[i,j]=0
		y_old=y
        print "Last SNR: "
	print SNR(y_old,y)
	return y

def SNR(A,B):
	if A.shape==B.shape:
		return numpy.sum(numpy.abs(A-B))/A.size
	else:
		raise Exception("Two matrices must have the same size!")

def palta(a, b):
        return 1 - (a - b)

def paltaex(a, b):
        return "1 - ( " + str(a) + " - " + str(b) + " )"

# This function is a symbolic version of delta function
# for this project.
def delta_sym(a, b):
        return 1 - (a - b)

def delta(a,b):
	if (a==b):
		return 1
	else:
		return 0

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
	print "The neighbers of (" + str(i) + ", " + str(j) + ")-th pixel are: " + str(neighbor)
	return neighbor

def neighbors_to_variables(var, neighbors):
        expression = "0 "
        for tuple in neighbors:
           expression = expression + " + "  + var + "_" + str(tuple[0]) + "_" + str(tuple[1]) + " "
        return expression


def cost_sym(y, x, y_old, index):
        alpha = 1
        beta = 10
        cost_sym = alpha * delta

def cost(y,x,y_old,index):
	alpha=1
	beta=10
        costt = alpha*delta(y,x)+ beta*sum(delta(y,y_old[i]) for i in index)
        costex = str(alpha) + " * " + str(paltaex(y, x))  + " + " + str(beta) + " * ( " + neighbors_to_variables('y', index)  + " )"
        print "cost expression: " + costex
        print "Cost of current pixel " + str(x)  +  " to be label " + str(y) + " with neighbors " + str(index) + " is " + str(costt)
	return costt

if __name__=="__main__":
	main()
