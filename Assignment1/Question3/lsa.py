from __future__ import print_function
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from os import path
import dataProcessing as dp
import testDataProcessing as tdp
import cosineSimilarity as cs
import numpy as np
import random
import os
import shutil
import operator
import shutil
import sys
import nltk
import re
import io

fileIndexMap = {}

def multiClassPerceptron(tfIdfMatrix, labelsMap, weights, fileIndexMap):
	'''
	Function that takes dataSet, labels, weights, bias, Epochs as parameters 
	and runs the Vanilla Perceptron algorithm and returns trained Weights and bias
	'''
	updated = True
	e = 0
	# Running the Perceptron for 100 Epochs
	while updated == True and e < 100:
		updated = False
		for i, row in enumerate(tfIdfMatrix):
			row = np.array(row)
			y = np.matmul(weights, np.transpose(row))
			maxVal = max(y)
			y = y.tolist()
			predictedLabel = y.index(maxVal)
			fileName = fileIndexMap[i]
			actualLabel = labelsMap[fileName]

			# Updates the weights when there is a misclassification
			if(str(actualLabel) != str(predictedLabel)):
				updated = True
				weights[int(predictedLabel)] -= row
				weights[int(actualLabel)] += row
		e += 1

	return weights

def makeFrequencyMatrix(bagOfWords):
	'''
	Function that makes the frequency matrix using the bag of words 
	'''
	mainMatrix = np.zeros([len(dp.fileAndDataMap), len(bagOfWords)])
	
	bagMap = {}
	for i, val in enumerate(bagOfWords):
		bagMap[val] = i

	# Looping through the FileAndDataMap and BagOfWords and 
	# updates the frequency in a matrix
	for file in dp.fileAndDataMap.keys():
		for word in dp.fileAndDataMap[file]:
			if word in bagMap:
				mainMatrix[dp.fileAndDataMap.keys().index(file)][bagMap[word]] += 1

	return mainMatrix

def getTfIdfMatrix(mainMatrix):
	'''
	Function that returns the Tf-Idf matrix from the frequency matrix created.
	'''
	listMatrix = mainMatrix.tolist()

	tfidfTrans = TfidfTransformer()
	tfIdfMatrix = tfidfTrans.fit_transform(listMatrix)
	tfIdfMatrix = tfIdfMatrix.toarray()

	# Augmenting the matrix with 1
	one = np.ones([len(tfIdfMatrix), 1])
	tfIdfMatrix = np.concatenate((tfIdfMatrix, one), axis=1)

	return tfIdfMatrix

def trainUsingPerceptron(tfIdfMatrix):	
	'''
	Function that takes the Tf-Idf Matrix and trains the data using 
	a multiclass Perceptron
	'''
	for i, key in enumerate(dp.fileAndDataMap):
		fileIndexMap[i] = key

	for key in dp.fileAndDataMap:
		dp.labelsMap[key] = key[key.rfind('/') - 1]

	# Initializing the weights
	weights = np.random.uniform(low=-1, high=1, size=(5, tfIdfMatrix.shape[1]))	
	weightsTrain = multiClassPerceptron(tfIdfMatrix, dp.labelsMap, weights, fileIndexMap)
	
	return weightsTrain, fileIndexMap

if __name__ == '__main__':

	trainDataPath = sys.argv[1]
	testDataPath = sys.argv[2]
	
	print("Splitting the data randomly into 80% Training and 20% Test Data")
	# dp.splitData(trainDataPath, testDataPath)
	print("Reading the Training data and cleaning (Stemming, Removed Stop words, etc)")
	uniqueBagOfWords = dp.readFullData(trainDataPath)
	print("Making the Frequency Matrix for Training data")
	mainMatrix = makeFrequencyMatrix(uniqueBagOfWords)
	print("Getting the Tf-Idf Matrix")
	tfIdfMatrix = getTfIdfMatrix(mainMatrix)
	print("Training the Perceptron on using this matrix")
	weights, fileIndexMap = trainUsingPerceptron(tfIdfMatrix)
	
	print("==================Training Done===================")

	print("Reading the Testing Data and cleaning (Stemming, Removed Stop words, etc)")
	tdp.readTestData(testDataPath)
	print("Making the Frequency Matrix for Test Data")
	mainTestMatrix = tdp.makeTestFrequencyMatrix(uniqueBagOfWords)
	print("Getting its corresponding Tf-Idf Matrix")
	testTfIdfMatrix = getTfIdfMatrix(mainTestMatrix)
	print("Using the Weights after Training for Testing")
	tdp.testUsingTrainedWeights(weights, testTfIdfMatrix)


	


