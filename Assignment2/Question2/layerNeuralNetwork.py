import numpy as np
import math
import trainingPenDigits as tpd
import sklearn.preprocessing as pre


def sigmoid(x):
    """
    Returns the sigmoid of x
    :param x: any input value
    :return: its sigmoid value
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    Return the tanh of x
    :param x:
    :return:
    """
    return np.tanh(x)


def derivativeTanh(x):
    """
    Returns the derivative of tanh of x
    :param x:
    :return:
    """
    derivatives = []

    for i in range(len(x)):
        derivatives.append((1 - (tanh(x[i]) * tanh(x[i]))))

    return np.array(derivatives)


def derivativeSigmoid(x):
    """
    :param x:
    :return:
    """
    return sigmoid(x) * (1 - sigmoid(x))


def softMax(x):
    """
    Compute softmax values for each value in x.
    """
    sMax = []
    for i in range(x.shape[0]):
        sMax.append(np.exp(x[i]) / np.sum(np.exp(x)))
    return np.array(sMax)


def calculateDerSoftMax(x):
    """
    Return the derivative of the softmax
    :param x:
    :return:
    """
    derivatives = []
    s = 0
    for i in range(x.shape[0]):
        s += np.exp(x[i])

    for i in range(x.shape[0]):
        derivatives.append(np.exp(x[i]) * (s - np.exp(x[i])) / (s * s))
    return np.array(derivatives)


def calculateError(sMax, labels):
    """
    Find Cross Entropy Error between the smax of output values
    and the actual labels
    :param sMax:
    :param labels:
    :return: error value
    """
    error = 0

    for i in [0, 1, 2]:
        error += (-1) * ((labels[i] * math.log(sMax[i])) +
                         (1 - labels[i]) * math.log(1 - sMax[i]))
    return error


def calculateDerCrossEntropy(sMax, labels):
    """
    Calculates the derivatives of cross entropy
    :param sMax:
    :param labels:
    :return: derivative of cross entropy
    """

    derivatives = np.zeros(sMax.shape[0])
    for i in range(sMax.shape[0]):
        derivatives[i] = -1 * ((labels[i] * (1 / sMax[i])) - (1 - labels[i]) * (1 / (1 - sMax[i])))

    return derivatives


def replicateRow(row, hiddenNeurons):
    """
    Replicates the row
    :param row:
    :return:
    """
    row = list(row)
    rows = [row] * hiddenNeurons
    return np.array(rows)


def calculateDerOutputWeights(hiddenActivation):
    """
    Calculates the derivative of the input to the output layer w.r.t weights
    :param hiddenActivation:
    :return:
    """
    derivatives = list(hiddenActivation)
    derivatives = [derivatives] * 3
    return np.array(derivatives)


def transformDataAndGetDataSets(actualData, labels, fold):
    """
    Transforming the data to make it easier for perceptron to classify
    and split the data into training and testing data sets
    """

    k = 5
    rows = int(math.floor(actualData.shape[0] / k))

    # Slicing the data into test and training data
    testData = actualData[fold * rows:fold * rows + rows, :]
    testLabels = labels[fold * rows:fold * rows + rows, :]

    trainingData = actualData[:fold * rows, :]
    trainLabels = labels[:fold * rows, :]

    trainingData = np.vstack([trainingData, actualData[fold * rows + rows:, :]])
    trainLabels = np.vstack([trainLabels, labels[fold * rows + rows:, :]])

    return trainingData, testData, trainLabels, testLabels


def trainWithVanillaNN(actualData, labelMatrix):
    """
    Trains using normal neural network which has sigmoid function
    as the activation function
    :param actualData: processed data
    :param labels: actual labels
    :return: weights and bias after training
    """

    # Initialize variables
    epoch = 40
    eta = 0.01
    inputNeurons = actualData.shape[1]
    hiddenNeurons = 10
    outputNeurons = 3
    k = 5

    accuracyList = []
    for j in range(k):

        # Initialize weights and bias of hidden layer and output layer
        weightsHidden = np.random.uniform(-0.75, 0.75, size=(inputNeurons + 1, hiddenNeurons))
        weightsOutput = np.random.uniform(-0.75, 0.75, size=(hiddenNeurons + 1, outputNeurons))

        fold = [j]
        trainingData, testData, trainLabels, testLabels = transformDataAndGetDataSets(actualData, labelMatrix, fold[0])
        biasInput = np.ones((trainingData.shape[0], 1))
        trainingData = np.append(trainingData, biasInput, axis=1)
        biasTest = np.ones((testData.shape[0], 1))
        testData = np.append(testData, biasTest, axis=1)
        print("Fold: ", fold)
        for i in range(epoch):
            for index, row in enumerate(trainingData):
                # weightsHidden = np.random.uniform(0.5, 0.5, size=(inputNeurons + 1, hiddenNeurons))
                # weightsOutput = np.random.uniform(0.5, 0.5, size=(hiddenNeurons + 1, outputNeurons))

                # Forward Propogation
                hiddenInputs = np.dot(row, weightsHidden)
                hiddenActivation = tanh(hiddenInputs)
                hiddenActivation = np.append(hiddenActivation, [1], axis=0)
                # hiddenActivation = sigmoid(hiddenInputs)
                # hiddenActivation = np.append(hiddenActivation, [1], axis=0)

                outputInputs = np.dot(hiddenActivation, weightsOutput)
                sMaxOutput = softMax(outputInputs)

                # Calculating stuff for backpropogation
                crossEntropyError = calculateError(sMaxOutput, trainLabels[index])

                # BackPropogation from Output to Hidden Layer
                derCrossEntropyErrors = calculateDerCrossEntropy(sMaxOutput, trainLabels[index])
                derSoftMaxOutputs = calculateDerSoftMax(outputInputs)
                derOutputWeights = hiddenActivation
                updateWeightsOutput = np.matmul(derOutputWeights.reshape(hiddenNeurons+1, 1),
                                                np.transpose(derCrossEntropyErrors * derSoftMaxOutputs).reshape(1, 3))

                # BackPropogation from Hidden to Input Layer
                derTotalError = np.array([])
                temp1 = derCrossEntropyErrors * derSoftMaxOutputs

                for p, eachWeight in enumerate(weightsOutput):
                    derTotalError = np.append(derTotalError, np.dot(np.transpose(temp1), eachWeight))

                derTotalError = derTotalError[:-1]
                # hiddenInputs = np.append(hiddenInputs, [1], axis=0)
                # derSigmoidHiddenInputs = derivativeTanh(hiddenInputs)
                derSigmoidHiddenInputs = derivativeTanh(hiddenInputs)

                temp2 = derTotalError * derSigmoidHiddenInputs

                replicatedInput = replicateRow(row, hiddenNeurons)

                updateWeightsInput = np.zeros(weightsHidden.shape)
                for p, eachRow in enumerate(np.transpose(replicatedInput)):
                    updateWeightsInput[p] = temp2 * eachRow

                weightsOutput = weightsOutput - eta * updateWeightsOutput
                weightsHidden = weightsHidden - eta * updateWeightsInput

                # print("RowNumber", index)
                # print("Row", row)
                # print("Hidden Input", hiddenInputs)
                # print("HiddenActivation", hiddenActivation)
                # print("OutputInput", outputInputs)
                # print("sMax", sMaxOutput)
                # print("Der Cross Entropy", derCrossEntropyErrors)
                # print("Der Soft max", derSoftMaxOutputs)
                # print("UpdateWeightsOutput", updateWeightsOutput)
                # print("Der Total Error", derTotalError)
                # print("Der Sigmoid HiddenInput", derSigmoidHiddenInputs)
                # print("myUpdatedWeightInput", updateWeightsInput)
                # print("WeightsOutput", weightsOutput)
                # print("WeightsHidden", weightsHidden)

        correct = 0
        for i, r in enumerate(testData):
            # Forward Propogation
            hiddenInputs = np.dot(r, weightsHidden)
            hiddenActivation = sigmoid(hiddenInputs)
            hiddenActivation = np.append(hiddenActivation, [1], axis=0)
            outputInputs = np.dot(hiddenActivation, weightsOutput)

            # Calculating stuff for backpropogation
            sMaxOutput = softMax(outputInputs)
            predictedLabel = np.argmax(sMaxOutput) + 1

            if predictedLabel == (np.argmax(testLabels[i])+1):
                correct += 1

            # print("Test Row", r)
            # print("RowNumber", i)
            # print("Hidden Input", hiddenInputs)
            # print("Hidden Activation", hiddenActivation)
            # print("OutputInputs", outputInputs)
            # print("Smax", sMaxOutput)
            # print("Predicted label ", predictedLabel)

        accuracy = float(correct) / testData.shape[0]
        accuracyList.append(accuracy)
        print("Accuracy: ", accuracy)
    print("Accuracy: ", sum(accuracyList) / 5)


def getDermatologyData():
    """
	Reading the Dermatology csv file and loading the data into np arrays
	"""
    path = raw_input("Enter the path of data: ")
    data = np.genfromtxt(path, delimiter=',', dtype='U')

    # Removing the rows with '?' in them
    char = '?'
    listToDelete = []
    for i, line in enumerate(range(len(data))):
        if char in data[i]:
            listToDelete.append(i)

    data = np.delete(data, listToDelete, axis=0)

    # Removing the rows with labels 4, 5, 6 since
    # we are considering only 1, 2, 3 labels
    data = data.astype(int)
    labels = data[:, -1]

    unwantedLabels = [4, 5, 6]
    listToDelete = []
    for i, line in enumerate(range(len(data))):
        if labels[i] in unwantedLabels:
            listToDelete.append(i)

    actualData = np.delete(data, listToDelete, axis=0)

    # Separating the labels and data into different arrays
    actualLabels = actualData[:, -1]
    actualData = actualData[:, :-1]

    actualData = pre.scale(actualData)

    # Change the label vector to label matrix
    # If Label is 2 then it becomes [0, 1, 0]
    labelMatrix = np.zeros((labels.shape[0], 3))
    for j in range(labels.shape[0]):
        if labels[j] == 1:
            labelMatrix[j][0] = 1
        if labels[j] == 2:
            labelMatrix[j][1] = 1
        if labels[j] == 3:
            labelMatrix[j][2] = 1

    trainWithVanillaNN(actualData, labelMatrix)

    return actualData, actualLabels


def getPenDigitData():
    """
    Read the data from csv file and load appropriately into arrays
    :return: Data and label arrays
    """
    path = raw_input("Enter the path of Train Data: ")
    data = np.genfromtxt(path, delimiter=',', dtype=int)
    labels = data[:, -1]

    unwantedLabels = [4, 5, 6, 7, 8, 9]
    listToDelete = []
    for i, line in enumerate(range(len(data))):
        if labels[i] in unwantedLabels:
            listToDelete.append(i)

    actualData = np.delete(data, listToDelete, axis=0)

    print(actualData.shape)
    # Separating the labels and data into different arrays
    actualLabels = actualData[:, -1]
    actualData = actualData[:, :-1]

    actualData = pre.scale(actualData)

    # Change the label vector to label matrix
    # If Label is 2 then it becomes [0, 1, 0]
    labelMatrix = np.zeros((actualLabels.shape[0], 4))
    for j in range(len(actualLabels)):
        if actualLabels[j] == 0:
            labelMatrix[j][0] = 1
        if actualLabels[j] == 1:
            labelMatrix[j][1] = 1
        if actualLabels[j] == 2:
            labelMatrix[j][2] = 1
        if actualLabels[j] == 3:
            labelMatrix[j][3] = 1

    # print(labelMatrix)
    testData, testLabels = tpd.loadTestData()

    tpd.trainUsingNNOnPenDigitData(actualData, labelMatrix, testData, testLabels)

    return actualData, actualLabels


if __name__ == '__main__':
    # Switch case simulation for choosing the dataSets
    dataSets = {
        1: getDermatologyData,
        2: getPenDigitData
    }

    choice = raw_input("\nEnter your choice of Data Set- \
						\n1. Dermatology Data \
						\n2. Pen Digit Data\n\n")

    actualData, labels = dataSets[int(choice)]()
