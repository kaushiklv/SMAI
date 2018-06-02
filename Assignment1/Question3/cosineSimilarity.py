from __future__ import print_function
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from os import path
from collections import Counter
import dataProcessing as dp
import testDataProcessing as tdp
import heapq
import itertools
import numpy as np
import lsa
import random
import os
import shutil
import operator
import shutil
import sys
import nltk
import re
import io
import sys

def trainUsingCosineSimilarity(trainTfIdfMatrix, testTfIdfMatrix, fileIndexMap, labelsMap, actualLabel):
	'''
	Function that predicts the label of the document using the cosine similarity.
	It takes trainTfIdfMatrix, testTfIdfMatrix, fileIndexMap, labelsMap, actualLabel as arguments.
	'''
	similarityList = []
	correctlyClassified = 0
	for index, row in enumerate(testTfIdfMatrix):
		# Gets the similarity values using cosine similarity
		similarityList = cosine_similarity(trainTfIdfMatrix, row.reshape(1, -1))
		simList = []
		for i in range(len(similarityList)):
			for j in range(len(similarityList[i])):
				simList.append(similarityList[i][j])

		# Gets the indices of the top 10 files with max similarity values
		top10SimilarFiles = heapq.nlargest(10, zip(simList, itertools.count()))
		indexList = []
		for item in top10SimilarFiles:
			indexList.append(item[1])

		# Display the Top 10 Files
		print("The Top 10 Similar Files - ")
		for ind in indexList:
			print("File " + str(ind+1) + ": " + fileIndexMap[ind])
		
		# Getting their labels
		labelList = []
		for i in indexList:
			fileName = fileIndexMap[i]
			labelList.append(labelsMap[fileName])

		# Get the maximum occuring label
		cnt = Counter(labelList)
		predictedLabel = cnt.most_common(1)	

		# Display the result
		print("Predicted Label: " + str(predictedLabel[0][0]) + " Actual Label: " + str(actualLabel))
		if str(predictedLabel[0][0]) == str(actualLabel):
			correctlyClassified += 1

	# Calculate and show the accuracy
	accuracy = float(correctlyClassified) / len(testTfIdfMatrix)
	print("Cosine Accuracy: ", accuracy*100)

if __name__ == '__main__':
	
	trainPath = sys.argv[1]
	testDocumentPath = sys.argv[2]
	testDocActualLabel = sys.argv[3]

	uniqueBagOfWords = dp.readFullData(trainPath)
	mainMatrix = lsa.makeFrequencyMatrix(uniqueBagOfWords)
	tfIdfMatrix = lsa.getTfIdfMatrix(mainMatrix)

	tdp.readTestDocument(testDocumentPath)
	mainTestMatrix = tdp.makeTestFrequencyMatrix(uniqueBagOfWords)
	testTfIdfMatrix = lsa.getTfIdfMatrix(mainTestMatrix)

	fileIndexMap = {}
	labelsMap = {}
	for i, key in enumerate(dp.fileAndDataMap):
		fileIndexMap[i] = key	
	for key in dp.fileAndDataMap:
		labelsMap[key] = key[key.rfind('/') - 1]

	trainUsingCosineSimilarity(tfIdfMatrix, testTfIdfMatrix, fileIndexMap, labelsMap, testDocActualLabel)



