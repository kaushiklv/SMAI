import numpy as np
import math
import random

def votedPerceptron(dataSet, labels, weights, bias, Epochs):
	'''
	Function that takes dataSet, labels, weights, bias, Epochs as parameters 
	and runs the Voted Perceptron algorithm and returns trained Weights and bias
	'''
	n = dataSet.shape[1]
	c = 1
	wList = []
	bList = []
	cList = []
	for e in range(Epochs):
		for i in (range(len(dataSet))):
			if (np.dot(weights, dataSet[i]) + (labels[i] * bias)) <= 0:
				wList.append(weights)
				bList.append(bias)
				cList.append(c)
				weights = weights + dataSet[i]
				bias = bias + labels[i]				
				c = 1				
			else:
				c += 1
		wList.append(weights)
		bList.append(bias)
		cList.append(c)																																																																																																																																																																												

	return wList, bList, cList

def getAccuracyVotedTestPerceptron(dataSet, labels, wList, bList, cList):
	'''
	Function that takes dataSet, labels, wList, bList, cList as parameters and 
	returns the accuracy of the classifier 
	'''
	y = 0
	correctClassified = 0
	misClassified = 0

	for i in range(len(dataSet)):
		for j in range(len(wList)):	
			y += cList[j] * np.sign(np.dot(dataSet[i], wList[j]) + bList[j])
		if(y > 0):
			correctClassified += 1
		else:
			misClassified += 1

	accuracy = float(correctClassified) / len(labels)
	return accuracy

