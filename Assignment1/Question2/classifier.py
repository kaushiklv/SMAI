import numpy as np
import math
import random
import matplotlib.pyplot as mpl
from numpy import genfromtxt
from numpy import linalg
import leastSquare as ls
import flda 

def getDataFromSet1():
	'''
	Reading the Ionosphere csv file and loading the data into np arrays
	'''	
	path = raw_input("Enter the path of data: ")
	data = np.genfromtxt(path, delimiter=',', dtype=int)
	actualData = data[:, :-1]
	labels = data[:,-1]

	return actualData, labels

def getDataFromSet2():
	'''
	Reading the Ionosphere csv file and loading the data into np arrays
	'''	
	path = raw_input("Enter the path of data: ")
	data = np.genfromtxt(path, delimiter=',', dtype=float)
	actualData = data[:, :-1]
	labels = data[:,-1]

	return actualData, labels

def compareBoth(actualData, labels):
	# Call Least Square Method and get the results
	lsWeights = ls.leastSquareMethod(actualData, labels)
	
	x = range(-3, 4)
	y = range(-3, 4)

	w0 = lsWeights[0]
	w1 = lsWeights[1]
	w2 = lsWeights[2]

	# Generating the points used to plot the line using the weights
	p1 = [0, -w2 / w1]
	p2 = [-w2 / w0, 0]
	p3 = [4, (-w2 - 4*w0) / w1]
	p4 = [(-w2 - 4*w1) / w0, 4]

	# Call Fischer LDA Method to get the weights, bias and projection Data
	weightsTrain, biasTrain, fullProjectedData = flda.fischerLDA(actualData, labels)

	class1Data = actualData[:len(actualData)/2, :]
	class2Data = actualData[len(actualData)/2:, :]

	w3 = weightsTrain[0]
	w4 = weightsTrain[1]
	w5 = biasTrain

	# Generating the points used to plot the line using the weights and bias
	p5 = [0, -w5 / w4]
	p6 = [-w5 / w3, 0]
	p7 = [3, (-w5 - 3*w3) / w4]
	p8 = [(-w5 - 3*w4) / w3, 3]

	# Plotting the data classes 
	mpl.scatter(class1Data[:,0], class1Data[:,1], c="blue", label="Class1")
	mpl.scatter(class2Data[:,0], class2Data[:,1], c="red", label="Class2")

	# Plotting the two classifiers and the projected line
	mpl.plot([p1[0], p2[0], p3[0], p4[0]], [p1[1], p2[1], p3[1], p4[1]], label="Least Square Classifier")
	mpl.plot([p5[0], p6[0], p7[0], p8[0]], [p5[1], p6[1], p7[1], p8[1]], label="Fischer Classifier")
	mpl.plot(fullProjectedData[:,0], fullProjectedData[:,1], label="Fischer Projection")
	mpl.title("Least Square vs Fischer's LDA")
	mpl.legend(loc="best")
	mpl.show()

	return weightsTrain, biasTrain, fullProjectedData

if __name__ == "__main__":
	# Switch case simulation for choosing the dataSets
	dataSets = {
		1: getDataFromSet1,
		2: getDataFromSet2,
	}

	choice = raw_input("\nEnter your choice of Data Set- \
						\n1. Data Set 1 \
						\n2. Data Set 2\n\n")

	actualData, labels = dataSets[int(choice)]()

	# Switch case simulation for choosing the method of classification
	methods = {
		1: ls.leastSquareHandle,
		2: flda.fischerLDAHandle,
		3: compareBoth
	}

	choice = raw_input("\nEnter the type of method - \
						\n1. Least Square Method \
						\n2. Fischer's LDA Method \
						\n3. Compare Both Methods\n\n")

	weightsTrain, biasTrain, fullProjectionData = methods[int(choice)](actualData, labels)
