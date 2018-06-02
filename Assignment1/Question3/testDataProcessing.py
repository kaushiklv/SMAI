from __future__ import print_function
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from os import path
import numpy as np
import dataProcessing as dp
import random
import os
import shutil
import operator
import shutil
import sys
import nltk
import re
import io

ps = PorterStemmer()

testFileDataMap = {}
testLabelsMap = {}
testFileIndexMap = {}

def testAndGetAccuracy(row, i, weights, testFileIndexMap):
	'''
	Function to Get the accuracy after testing using the weights
	'''
	row = np.array(row)

	y = np.matmul(weights, np.transpose(row))
	maxVal = max(y)
	y = y.tolist()
	predictedLabel = y.index(maxVal)
	fileName = testFileIndexMap[i]
	actualLabel = testLabelsMap[fileName]

	if(str(actualLabel) == str(predictedLabel)):
		return True
	else:
		return False


def testUsingTrainedWeights(weights, testTfIdfMatrix):
	'''
	Function which uses testAndGetAccuracy function to finally calculate the accuracy
	'''
	for i, key in enumerate(testFileDataMap):
		testFileIndexMap[i] = key

	for key in testFileIndexMap.keys():
		testLabelsMap[testFileIndexMap[key]] = testFileIndexMap[key][testFileIndexMap[key].rfind('/') - 1]

	correctlyClassified = 0

	# Looping through the Test Tf-Idf Matrix to get accuracy
	for i, row in enumerate(testTfIdfMatrix):
		if(testAndGetAccuracy(row, i, weights, testFileIndexMap)):
			correctlyClassified += 1

	accuracy = float(correctlyClassified) / testTfIdfMatrix.shape[0]
	print("Accuracy of Perceptron: ", accuracy * 100) 


def makeTestFrequencyMatrix(bagOfWords):
	'''
	Makes the frequency matrix of the test data 
	'''
	mainMatrix = np.zeros([len(testFileDataMap), len(bagOfWords)])

	bagMap = {}
	for i, val in enumerate(bagOfWords):
		bagMap[val] = i

	for file in testFileDataMap.keys():
		for word in testFileDataMap[file]:
			if word in bagMap:
				mainMatrix[testFileDataMap.keys().index(file)][bagMap[word]] += 1

	return mainMatrix


def readTestFilesUtil(path):
	'''
	Recursive function to read the test data and make 
	map of file and its list of words after processing them
	'''
	fileList = os.listdir(path)

	for file in fileList:
		if os.path.isdir(path + "/" + file):
			readTestFilesUtil(path + "/" + file)
		else:
			fileData = []
			tokenizedLines = []
			with io.open(path + "/" + file,'r',encoding='ascii',errors='ignore') as f:
				for line in f:
					tokens = word_tokenize(line)			
					tokenizedLines.append([ps.stem(word).encode('ascii','ignore').lower() for word in tokens if len(word) >= 2])

			testBagOfWords = [word for line in tokenizedLines for word in line]
			nonUniqueTestBag = [w for w in testBagOfWords if not w in dp.stopWords]
			testFileDataMap[path + "/" + file] = nonUniqueTestBag


def readTestDocument(path):
	'''
	Reads the data of the file and stores its words after processing
	'''
	fileData = []
	tokenizedLines = []
	with io.open(path,'r',encoding='ascii',errors='ignore') as f:
		for line in f:
			tokens = word_tokenize(line)			
			tokenizedLines.append([ps.stem(word).encode('ascii','ignore').lower() for word in tokens if len(word) >= 2])

	testBagOfWords = [word for line in tokenizedLines for word in line]
	nonUniqueTestBag = [w for w in testBagOfWords if not w in dp.stopWords]
	testFileDataMap[path] = nonUniqueTestBag


def readTestData(path):
	'''
	Handler function to read the test data
	'''
	readTestFilesUtil(path)
