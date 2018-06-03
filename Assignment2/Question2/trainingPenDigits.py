import numpy as np
import sklearn.preprocessing as pre
import layerNeuralNetwork as nn


def loadTestData():
    """
    Loads the test data into the array
    :param path:
    :return: testData, labels
    """
    path = raw_input("Enter the path of Test Data: ")
    data = np.genfromtxt(path, delimiter=',', dtype=int)

    labels = data[:, -1]

    unwantedLabels = [4, 5, 6, 7, 8, 9]
    listToDelete = []
    for i, line in enumerate(range(len(data))):
        if labels[i] in unwantedLabels:
            listToDelete.append(i)

    actualData = np.delete(data, listToDelete, axis=0)

    # print(actualData.shape)
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

    return actualData, actualLabels


def trainUsingNNOnPenDigitData(actualData, labels, testData, testLabels):
    """
    Take the actual Data and labels and train using neural network
    :return:
    """
    # Initialize variables
    epoch = 40
    eta = 0.01
    inputNeurons = actualData.shape[1]
    hiddenNeurons = 10
    outputNeurons = 4

    # Initialize weights and bias of hidden layer and output layer
    weightsHidden = np.random.uniform(-1, 0.99, size=(inputNeurons+1, hiddenNeurons))
    weightsOutput = np.random.uniform(-1, 0.99, size=(hiddenNeurons+1, outputNeurons))

    biasInput = np.ones((actualData.shape[0], 1))
    actualData = np.append(actualData, biasInput, axis=1)
    biasTest = np.ones((testData.shape[0], 1))
    testData = np.append(testData, biasTest, axis=1)

    # Outer loop for all the Epochs
    for i in range(epoch):
        print("Epoch: ", i)
        for index, row in enumerate(actualData):

            # Forward Propogation
            hiddenInputs = np.dot(row, weightsHidden)
            # hiddenActivation = nn.sigmoid(hiddenInputs)
            hiddenActivation = nn.sigmoid(hiddenInputs)
            hiddenActivation = np.append(hiddenActivation, [1], axis=0)
            outputInputs = np.dot(hiddenActivation, weightsOutput)
            sMaxOutput = nn.softMax(outputInputs)

            # Calculating stuff for backpropogation
            crossEntropyError = nn.calculateError(sMaxOutput, labels[index])

            # BackPropogation from Output to Hidden Layer
            derCrossEntropyErrors = nn.calculateDerCrossEntropy(sMaxOutput, labels[index])
            derSoftMaxOutputs = nn.calculateDerSoftMax(outputInputs)
            derOutputWeights = hiddenActivation
            updateWeightsOutput = np.matmul(derOutputWeights.reshape(hiddenNeurons+1, 1),
                                            np.transpose(derCrossEntropyErrors * derSoftMaxOutputs).reshape(1, 4))

            # BackPropogation from Hidden to Input Layer
            derTotalError = np.array([])
            temp1 = derCrossEntropyErrors * derSoftMaxOutputs

            for p, eachWeight in enumerate(weightsOutput):
                derTotalError = np.append(derTotalError, np.dot(np.transpose(temp1), eachWeight))

            derTotalError = derTotalError[:-1]
            derSigmoidHiddenInputs = nn.derivativeSigmoid(hiddenInputs)
            temp2 = derTotalError * derSigmoidHiddenInputs

            replicatedInput = nn.replicateRow(row, hiddenNeurons)

            updateWeightsInput = np.zeros(weightsHidden.shape)
            for p, eachRow in enumerate(np.transpose(replicatedInput)):
                updateWeightsInput[p] = temp2 * eachRow

            weightsOutput = weightsOutput - eta * updateWeightsOutput
            weightsHidden = weightsHidden - eta * updateWeightsInput

    correct = 0
    for j, r in enumerate(testData):
        # Forward Propogation
        hiddenInputs = np.dot(r, weightsHidden)
        # hiddenActivation = nn.sigmoid(hiddenInputs)
        hiddenActivation = nn.tanh(hiddenInputs)
        hiddenActivation = np.append(hiddenActivation, [1], axis=0)
        outputInputs = np.dot(hiddenActivation, weightsOutput)

        # Calculating stuff for backpropogation
        sMaxOutput = nn.softMax(outputInputs)
        predictedLabel = np.argmax(sMaxOutput) + 1
        if predictedLabel == (np.argmax(testLabels[j]) + 1):
            correct += 1

    print("Correct ", correct)
    print("Total ", testData.shape[0])
    accuracy = float(correct) / testData.shape[0]
    print("Accuracy: ", accuracy)
