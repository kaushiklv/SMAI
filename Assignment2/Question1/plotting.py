import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"


def plotResultsLearningRate():
    """
    Plot the results when learning rate is varied
    :return:
    """
    x = [0.0001, 0.001, 0.01, 0.1, 1]
    y = [1.1121616285324096, 1.1499698184967042, 1.1390595510482788, 1.1565211568832396, 1.2]

    plt.plot(x, y)
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel('Learning Rates')
    plt.ylabel('Accuracy')
    plt.title('Learning Rate vs Accuracy')
    plt.savefig('learningRate.jpg')
    plt.show()


def plotResultsBatchSize():
    """
    Plot the results when batch-size is varied
    :return:
    """
    x = [5, 50, 500, 5000]
    y = [0.6272, 0.5958, 0.6256, 0.2143]

    plt.plot(x, y)
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy')
    plt.title('Batch Size vs Accuracy')
    plt.savefig('batchSize.jpg')
    plt.show()


def plotResultsConvolutionFilters():
    """
    Plot the results when number of convolution filters are changed
    :return:
    """
    x = [3, 6, 10, 16, 160]
    y = [0.546, 0.579, 0.6181, 0.6494, 0.6592]

    plt.figure(1)
    plt.plot(x, y)
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel('No Of Conv Filters Layer 1')
    plt.ylabel('Accuracy')
    plt.title('Conv Filters vs Accuracy (Layer 2 Conv Filters fixed at 16)')
    plt.savefig('convFil1.jpg')
    plt.show()

    z = [3, 6, 10, 50, 100]
    p = [0.5008, 0.5675, 0.5787, 0.6336, 0.6403]

    plt.figure(2)
    plt.plot(z, p)
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel('No Of Conv Filters Layer 2')
    plt.ylabel('Accuracy')
    plt.title('Conv Filters vs Accuracy (Layer 1 Conv Filters fixed at 6)')
    plt.savefig('convFil2.jpg')
    plt.show()


def plotResultsAccuracy():
    """
    Plots the accuracy of the neural network in different settings
    :return:
    """
    epochs = [5, 10, 20, 30, 40]
    sigmoidAccuraciesDerm = [47.91, 58.33, 70.83, 62.50, 52.08]
    tanhAccuraciesDerm = [48.75, 50.00, 47.91, 52.91, 45.00]

    sigmoidAccuraciesPen = [27.12, 28.66, 24.38, 25.51, 24.73]
    tanhAccuraciesPen = [27.68, 35.38, 32.66, 24.60, 25.29]

    plt.plot(epochs, sigmoidAccuraciesDerm)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Dermatology Sigmoid - Epochs vs Accuracy')
    plt.savefig('dermSig.jpg')
    plt.show()

    plt.plot(epochs, tanhAccuraciesDerm)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Dermatology Tanh - Epochs vs Accuracy')
    plt.savefig('dermTanh.jpg')
    plt.show()

    plt.plot(epochs, sigmoidAccuraciesPen)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Pen Digit Sigmoid - Epochs vs Accuracy')
    plt.savefig('penSig.jpg')
    plt.show()

    plt.plot(epochs, tanhAccuraciesPen)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Pen Digit Tanh - Epochs vs Accuracy')
    plt.savefig('penTanh.jpg')
    plt.show()


if __name__ == "__main__":
    plotResultsLearningRate()
    plotResultsBatchSize()
    plotResultsConvolutionFilters()
    plotResultsAccuracy()
