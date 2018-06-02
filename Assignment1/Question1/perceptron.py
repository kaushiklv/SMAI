import numpy as np
import math
import random
import vanillaPerceptron as VaP
import votedPerceptron as VoP
import matplotlib.pyplot as mpl
from numpy import genfromtxt

def getIonosphereData():
	'''
	Reading the Ionosphere csv file and loading the data into np arrays
	'''	
	path = raw_input("Enter the path of data: ")
	data = np.genfromtxt(path, delimiter=',', dtype=float)
	actualData = data[:, :-1]

	data = np.genfromtxt(path, delimiter=',', dtype='U')
	labels = data[:,-1]

	# Changing the labels into 1 and -1 to ease classification
	labels[labels == 'g'] = 1
	labels[labels == 'b'] = -1

	# Changing the type of data from Unicode to Integer
	np.set_printoptions(precision=5, suppress=True)
	labels = np.genfromtxt(labels)

	return actualData, labels

def getBreastCancerData():
	'''
	Reading the Breast Cancer csv file and loading the data into np arrays
	'''	
	path = raw_input("Enter the path of data: ")
	data = np.genfromtxt(path, delimiter=',', dtype='U')
	actualData = data[:, 1:-1]

	# Deleting the rows with missing data from the dataSet
	char = '?'
	listToDelete = []
	for i, line in enumerate(range(len(actualData))):
		if char in actualData[i]:
			listToDelete.append(i)

	actualData = np.delete(actualData, listToDelete, axis=0)

	# Converting the data into int type from Unicode
	dataList = []
	for line in actualData:
		lineItem = map(int, line)
		dataList.append(lineItem)

	ActualData = np.array(dataList)
	
	# Changing the labels from 2 and 4 to 1 and -1 to ease classification
	labels = map(int, data[:,-1])
	labels[labels == 2] = 1
	labels[labels == 4] = -1

	return ActualData, labels

def transformDataAndGetDataSets(actualData, labels, fold):
	'''
	Transforming the data to make it easier for perceptron to classify
	and split the data into training and testing data sets
	'''
	# Transforming data of -1 class to 1 by mutltipying the whole row by -1
	# This eases classification
	for i, label in enumerate(labels):
		if label < 0:
			actualData[i, :] *= -1

	k = 10
	rows = int(math.floor(actualData.shape[0] / k))

	# Slicing the data into test and training data	
	testData = actualData[fold*rows:fold*rows + rows, :]
	testLabels = labels[fold*rows:fold*rows + rows]

	trainingData = actualData[:fold*rows, :]
	trainLabels = labels[:fold*rows]

	trainingData = np.vstack([trainingData, actualData[fold*rows + rows:, :]])
	trainLabels = np.append(trainLabels, labels[fold*rows + rows:])

	return trainingData, testData, trainLabels, testLabels

def plotResults(Epochs, teVaList, teVoList):
	'''
	Plotting the results of accuracies of Vanilla and Voted Perceptrons
	versus the Epochs
	'''
	mpl.suptitle("Vanilla vs Voted")
	mpl.plot(Epochs, teVaList, c="red",label="Vanilla")
	mpl.plot(Epochs, teVoList, c="blue",label="Voted")
	mpl.xlabel('Epochs')
	mpl.ylabel('Accuracy')
	mpl.legend(loc="best")
	mpl.show()

if __name__ == '__main__':
	
	# Switch case simulation for choosing the dataSets
	dataSets = {
		1: getIonosphereData,
		2: getBreastCancerData,
	}

	choice = raw_input("\nEnter your choice of Data Set- \
						\n1. Ionosphere Data \
						\n2. Breast Cancer Data\n\n")

	actualData, labels = dataSets[int(choice)]()
	
	k = 10
	Epochs = range(10, 55, 5)
	teVaList = []	
	teVoList = []
	# folds = random.sample(range(0, 10), 10)

	# Outer loop for all the Epochs
	for e in Epochs:
		temp1 = []
		temp2 = []
		avg = 0

		# Inner loop for the number folds which is 10 in this case
		for i in range(k):

			# Picking a random fold
			# fold = folds[i]	
			fold = random.randrange(0, 10)
			trainingData, testData, trainLabels, testLabels = transformDataAndGetDataSets(actualData, labels, fold)
			
			n = trainingData.shape[1]
			weights = np.zeros(n)
			bias = 0

			# Run the vanilla perceptron for the selected epochs and append accuracies
			weightsTrain, biasTrain = VaP.vanillaPerceptron(trainingData, trainLabels, weights, bias, e)
			testVanillaOutput = VaP.getAccuracyVanillaPerceptron(testData, testLabels, weightsTrain, biasTrain)
			temp1.append(testVanillaOutput)

			# Run the voted perceptron for the selected epochs and append accuracies
			wList, bList, cList = VoP.votedPerceptron(trainingData, trainLabels, weights, bias, e)
			testVotedOutput = VoP.getAccuracyVotedTestPerceptron(testData, testLabels, wList, bList, cList)
			temp2.append(testVotedOutput)

		# Taking average of accuracies for the particular Epoch
		avg1 = float(sum(temp1) / len(temp1))
		teVaList.append(avg1*100)
			
		avg2 = float(sum(temp2) / len(temp2))
		teVoList.append(avg2*100)
	
	# Printing the Results and Plotting them
	print teVaList
	print teVoList
	print Epochs
	plotResults(Epochs, teVaList, teVoList)

	