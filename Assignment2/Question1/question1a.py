import numpy as np
import cv2
from PIL import Image


def ReLU(x):
    """
    Compute the relu
    :param x:
    :return:
    """
    return np.where(x > 0, 1.0, 0.0)


def softMax(x):
    """
    Compute softmax values for each value in x.
    """
    sMax = []
    for i in range(x.shape[0]):
        sMax.append(np.exp(x[i]) / np.sum(np.exp(x)))
    return np.array(sMax)


def convolutionLayer1(image, filters):
    """
    Convolves the image by applying the filter
    :param image:
    :param filters:
    :return: convolutionMatrix
    """
    convolutionMatrix = np.zeros((28, 28, 6))

    for fil in range(6):
        for rowStride in range(image.shape[0] - filters.shape[0] + 1):
            for colStride in range(image.shape[0] - filters.shape[0] + 1):
                convolutionMatrix[rowStride][colStride][fil] = \
                    np.sum(filters * image[rowStride:rowStride + 5,
                                           colStride:colStride + 5, :])

    return convolutionMatrix


def maxPoolLayer1(convolutionMatrix):
    """
    Performs max pooling on the convoluted matrix
    :param convolutionMatrix:
    :return: max pooled matrix
    """
    maxPoolMatrix = np.zeros((14, 14, 6))

    for fil in range(6):
        for rowStride in range(0, 28, 2):
            for colStride in range(0, 28, 2):
                maxPoolMatrix[rowStride / 2][colStride / 2][fil] = \
                        np.max(convolutionMatrix[rowStride:rowStride + 1,
                                                 colStride:colStride + 1, :])

    return maxPoolMatrix


def convolutionLayer2(maxPoolMatrix, filters):
    """
    Convolves the given matrix by applying the filter
    :param maxPoolMatrix1:
    :param filters2:
    :return: convoluted matrix
    """
    convolutionMatrix = np.zeros((10, 10, 16))

    for fil in range(16):
        for rowStride in range(maxPoolMatrix.shape[0] - filters.shape[0] + 1):
            for colStride in range(maxPoolMatrix.shape[0] - filters.shape[0] + 1):
                convolutionMatrix[rowStride][colStride][fil] = \
                    np.sum(filters * maxPoolMatrix[rowStride:rowStride + 5,
                                                   colStride:colStride + 5, :])

    return convolutionMatrix


def maxPoolLayer2(convolutionMatrix):
    """
    Perform max pooling on the given convolution matrix
    :param convolutionMatrix:
    :return: max pooled matrix
    """
    maxPoolMatrix = np.zeros((5, 5, 16))

    for fil in range(16):
        for rowStride in range(0, 10, 2):
            for colStride in range(0, 10, 2):
                maxPoolMatrix[rowStride / 2][colStride / 2][fil] = \
                        np.max(convolutionMatrix[rowStride:rowStride + 1,
                                                 colStride:colStride + 1, :])

    return maxPoolMatrix


def forwardPassOfNeuralNetwork(inputs):
    """
    Perform the forward pass of the neural network
    :param inputs:
    :return:
    """
    # Initializing variables
    hiddenLayer1 = 120
    hiddenLayer2 = 84
    outputLayer = 10

    weightsHiddenLayer1 = np.random.uniform(-0.1, 0.1, size=(hiddenLayer1, inputs.shape[0]))
    weightsHiddenLayer2 = np.random.uniform(-0.1, 0.1, size=(hiddenLayer2, hiddenLayer1))
    weightsOutputLayer = np.random.uniform(-0.1, 0.1, size=(outputLayer, hiddenLayer2))

    # Forward Pass of Neural Network
    hiddenLayer1Output = np.matmul(weightsHiddenLayer1, inputs)
    hiddenLayer2Output = np.matmul(weightsHiddenLayer2, hiddenLayer1Output)
    outputLayerOutput = np.matmul(weightsOutputLayer, hiddenLayer2Output)

    # Performing softmax of the output
    softMaxOutputs = softMax(outputLayerOutput)
    print(softMaxOutputs)


if __name__ == "__main__":
    imagePath = raw_input("Enter the path of the image: ")
    # image = cv2.imread(imagePath)
    image = Image.open(imagePath)

    # Resize the image to 32 x 32 x 3
    imageResized = image.resize((32, 32), Image.ANTIALIAS)
    imageResized = np.array(imageResized)

    img = Image.fromarray(imageResized, 'RGB')
    img = img.resize((312, 312))
    img.save('newPic.jpg')
    img.show()

    filters1 = np.random.randn(5, 5, 3)

    # Perform the First convolution
    convolutionMatrix1 = convolutionLayer1(imageResized, filters1)

    # Display the intermediate image
    img1 = Image.fromarray(convolutionMatrix1, 'RGB')
    img1 = img1.resize((312, 312))
    img1.show()

    # Apply the ReLU function on the convolution outputs
    reluMatrix1 = ReLU(convolutionMatrix1)

    # Perform the First Max Pooling
    maxPoolMatrix1 = maxPoolLayer1(reluMatrix1)

    # Display the intermediate image
    img2 = Image.fromarray(maxPoolMatrix1, 'RGB')
    img2 = img2.resize((312, 312))
    img2.show()

    # Perform the Second convolution
    filters2 = np.random.randn(5, 5, 6)
    convolutionMatrix2 = convolutionLayer2(maxPoolMatrix1, filters2)

    # Display the intermediate image
    img3 = Image.fromarray(convolutionMatrix2, 'RGB')
    img3 = img3.resize((312, 312))
    img3.show()

    # Apply ReLU function on the second convolution
    reluMatrix2 = ReLU(convolutionMatrix2)

    # Perform the Second Max Pooling
    maxPoolMatrix2 = maxPoolLayer2(convolutionMatrix2)

    # Display the intermediate image
    img4 = Image.fromarray(maxPoolMatrix2, 'RGB')
    img4 = img4.resize((312, 312))
    img4.show()

    # Perform the forward pass of the Neural Network
    inputs = maxPoolMatrix2.flatten()
    forwardPassOfNeuralNetwork(inputs)
