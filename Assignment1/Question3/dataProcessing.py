from __future__ import print_function
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from os import path
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

# Global Maps and Lists Needed
fileAndDataMap = {}
stopWords = set(stopwords.words('english'))
words = []
fileList = []
fileListLengths = []
frequencyMatrix = np.array([])
fileNameMap = {}
labelsMap = {}

ps = PorterStemmer()

def getFilesList(path):
	l = os.listdir(path)
	tempList = []
	for ele in l:
		if os.path.isdir(path + "/" + ele):			# Recursive call if it is a directory
			getFilesList(path + "/" + ele)		
		else:
			fileList.append(ele)					# Add to the map if it is a file
			tempList.append(ele)
		fileNameMap[path] = tempList
			
def splitData(src, dest):
	'''
	Function which randomly splits the data into 80% and 20% 
	'''
	getFilesList(path)

	src = "/media/kaushik/Studies/IIITH/SecondSem/SMAI/Assignment1/Question3/q3data/train"
	dest = "/media/kaushik/Studies/IIITH/SecondSem/SMAI/Assignment1/Question3/q3data/test"

	del fileNameMap[src]
	for key in fileNameMap:
		randomNos = random.sample(range(0, len(fileNameMap[key])), int(0.2 * len(fileNameMap[key])))
		for i in randomNos:
			shutil.move(src + "/" + key[-1] + "/" + fileNameMap[key][i], dest + "/" + key[-1] + "/" + fileNameMap[key][i]) 

def readFilesUtil(path):
	'''
	Function that recursively reads the data from path
	and makes the Bag Of Words and map of file and its words
	After processing the words
	'''
	fileList = os.listdir(path)

	for file in fileList:
		if os.path.isdir(path + "/" + file):
			readFilesUtil(path + "/" + file)
		else:
			fileData = []
			tokenizedLines = []
			with io.open(path + "/" + file,'r',encoding='ascii',errors='ignore') as f:
				for line in f:
					tokens = word_tokenize(line)			
					tokenizedLines.append([ps.stem(word).encode('ascii','ignore').lower() for word in tokens if len(word) >= 2])

			bagOfWords = [word for line in tokenizedLines for word in line]
			nonUniqueBag = [w for w in bagOfWords if not w in stopWords]

			cleanBagOfWords = set([w for w in bagOfWords if not w in stopWords])
			words.append(cleanBagOfWords)
			fileAndDataMap[path + "/" + file] = nonUniqueBag


def readFullData(path):
	'''
	Handler function to read the words
	'''
	readFilesUtil(path)
	uniqueBagOfWords = set([word for line in words for word in line])
	return uniqueBagOfWords