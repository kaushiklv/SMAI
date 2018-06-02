import numpy as np
import math
import random
import matplotlib.pyplot as mpl
from numpy import genfromtxt
from numpy import linalg

def leastSquareMethod(actualData, labels):
	'''
	Function that calculates the weights using Least square approach
	'''
	# Augmenting the data with 1
	n = actualData.shape[0]
	ones = np.ones(n)
	augData = np.column_stack([actualData, ones])

	# Calculating the Weights 
	weights = np.dot(np.dot(np.linalg.linalg.inv(np.dot(np.transpose(augData), augData)), np.transpose(augData)), labels)
	
	return weights

def plotResultsLeastSquare(actualData, weights):
	'''
	Function that Plots the data classes and the 
	least square classifier line.
	'''
	# Generating the points using the weights
	x = range(-3, 4)
	y = range(-3, 4)

	w0 = weights[0]
	w1 = weights[1]
	w2 = weights[2]

	p1 = [0, -w2 / w1]
	p2 = [-w2 / w0, 0]
	p3 = [4, (-w2 - 4*w0) / w1]
	p4 = [(-w2 - 4*w1) / w0, 4]

	xs, ys = actualData.T

	class1Xs = xs[:len(xs)/2]
	class2Xs = xs[len(xs)/2:]

	class1Ys = ys[:len(ys)/2]
	class2Ys = ys[len(ys)/2:]

	# Plotting the data from both classes
	mpl.scatter(class1Xs, class1Ys, c="blue", label="Class 1")
	mpl.scatter(class2Xs, class2Ys, c="red", label="Class 2")
	
	# Plotting the Least Square Classifier line
	mpl.plot([p1[0], p2[0], p3[0], p4[0]], [p1[1], p2[1], p3[1], p4[1]], label="Classifier")
	mpl.title("Least Square Method")
	mpl.legend(loc="best")
	mpl.show()

def leastSquareHandle(actualData, labels):
	'''
	Handler function which calls the Least square method
	And takes these weights and call the plot function
	'''
	weights = leastSquareMethod(actualData, labels)
	plotResultsLeastSquare(actualData, weights)

	return weights, labels, actualData

