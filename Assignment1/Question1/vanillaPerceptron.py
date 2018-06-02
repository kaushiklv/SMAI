import numpy as np
import math
import random

def vanillaPerceptron(dataSet, labels, weights, bias, Epochs):
	'''
	Function that takes dataSet, labels, weights, bias, Epochs as parameters 
	and runs the Vanilla Perceptron algorithm and returns trained Weights and bias
	'''
	for e in range(Epochs):
		for i in (range(len(dataSet))):
			if (np.dot(weights, dataSet[i]) + (labels[i] * bias)) <= 0:
				weights = weights + dataSet[i]
				bias = bias + labels[i]

	return weights, bias

def getAccuracyVanillaPerceptron(dataSet, labels, weights, bias):
	'''
	Function that takes dataSet, labels, weights, bias as parameters and 
	returns the accuracy of the classifier 
	'''
	y = 0
	correctClassified = 0
	misClassified = 0
	for i in range(len(dataSet)):
		y = np.dot(dataSet[i], weights) + bias
		if (y > 0):
			correctClassified += 1
		else:
			misClassified += 1

	accuracy = float(correctClassified) / len(labels)
	return accuracy
