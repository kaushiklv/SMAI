import numpy as np
import math
import random
import leastSquare as ls
import matplotlib.pyplot as mpl
from numpy import genfromtxt
from numpy.linalg import inv, norm

def vanillaPerceptron(dataSet, labels, weights, bias, Epochs):
	'''
	Function that takes dataSet, labels, weights, bias, Epochs as parameters 
	and runs the Vanilla Perceptron algorithm and returns trained Weights and bias
	'''
	for e in range(Epochs):
		for i in (range(len(dataSet))):
			if (labels[i] * (np.dot(weights, dataSet[i]) + bias)) <= 0:
				weights = weights + (np.dot(dataSet[i], labels[i])) 
				bias = bias + labels[i]

	return weights, bias

def plotResults(weightsTrain, biasTrain, class1Data, class2Data, fullProjectedData):
	'''
	Function that plots the data classes and the both the lines
	i.e. one line on which data is projected and another is the classifier.
	'''
	mpl.scatter(class1Data[:,0], class1Data[:,1], c="blue", label="Class1")
	mpl.scatter(class2Data[:,0], class2Data[:,1], c="red", label="Class2")
	mpl.plot(fullProjectedData[:,0], fullProjectedData[:,1], label="Fischer Projection")

	w0 = weightsTrain[0]
	w1 = weightsTrain[1]
	w2 = biasTrain

	p1 = [0, -w2 / w1]
	p2 = [-w2 / w0, 0]
	p3 = [3, (-w2 - 3*w0) / w1]
	p4 = [(-w2 - 3*w1) / w0, 3]

	mpl.plot([p1[0], p2[0], p3[0], p4[0]], [p1[1], p2[1], p3[1], p4[1]], label="Perceptron Classifier")
	mpl.legend(loc="best")
	mpl.show()

def fischerLDA(actualData, labels):
	'''
	Function that performs Fischer LDA and returns the weights and bias
	using which we can classify the data.
	'''
	# Splitting the data into individual classes
	class1Data = actualData[:len(actualData)/2, :]
	class2Data = actualData[len(actualData)/2:, :]

	# Calculate their means
	class1Means = np.mean(class1Data, axis = 0)
	class2Means = np.mean(class2Data, axis = 0)
	
	# Subtract Data from their mean
	class1DataMinusMean = class1Data - class1Means
	class2DataMinusMean = class2Data - class2Means

	# Find their individual Covariance
	class1Cov = np.cov(np.transpose(class1DataMinusMean)) * (class1Data.shape[0] - 1)
	class2Cov = np.cov(np.transpose(class2DataMinusMean)) * (class2Data.shape[0] - 1)

	# Add them to get the Covariance matrix
	covMatrix = class1Cov + class2Cov
	
	# Calculate the weights
	weights = np.dot(inv(covMatrix), (class1Means - class2Means))

	# Get the normalized weights
	normalizedWeights = weights / norm(weights)

	# Project the data onto a line represented by the weights
	# Get the magnitudes of the data using the weights
	class1MagVector = np.dot(class1Data, normalizedWeights)
	class2MagVector = np.dot(class2Data, normalizedWeights)

	class1ProjectedData = []
	class2ProjectedData = []

	# Getting the projected points using the magnitudes and weights
	for mag in class1MagVector:
		class1ProjectedData.append(np.dot(mag, normalizedWeights))

	for mag in class2MagVector:
		class2ProjectedData.append(np.dot(mag, normalizedWeights))

	class1ProjectedData = np.asarray(class1ProjectedData)
	class2ProjectedData = np.asarray(class2ProjectedData)

	# Combine to get the full projected Data
	fullProjectedData = class1ProjectedData
	fullProjectedData = np.vstack([fullProjectedData, class2ProjectedData])

	initWeights = np.zeros(fullProjectedData.shape[1])
	initBias = 1

	# Train the perceptron to classify these projected points
	weightsTrain, biasTrain = vanillaPerceptron(fullProjectedData, labels, initWeights, initBias, 50)

	return weightsTrain, biasTrain, fullProjectedData

def fischerLDAHandle(actualData, labels):
	'''
	Handler function which calls the Fischer LDA function and gets the
	weights, bias and the projected data. Now it calls the plot method
	which plos the results
	'''
	weightsTrain, biasTrain, fullProjectedData = fischerLDA(actualData, labels)
	class1Data = actualData[:len(actualData)/2, :]
	class2Data = actualData[len(actualData)/2:, :]

	plotResults(weightsTrain, biasTrain, class1Data, class2Data, fullProjectedData)
	return weightsTrain, biasTrain, fullProjectedData
	

	


