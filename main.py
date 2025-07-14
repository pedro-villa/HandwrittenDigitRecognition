import numpy as np
import tensorflow as tf
from tensorflow import keras

def getData(gridSize):
    pass

def RELU(x): 
    return np.maximum(x, 0)

def loadMNIST():
    (data_train, labels_train), (data_test, label_test) = keras.datasets.mnist.load_data()
    # Normalize data
    data_train = data_train.astype('float32')/255
    data_test = data_test.astype('float32')/255
    # Reshape array
    data_train = data_train.reshape(-1, 28*28)
    data_test = data_test.reshape(-1, 28*28)

    return (data_train, labels_train), (data_test, label_test)

def evaluate(data, nLayers, weights, biases):
    layerOutput = data
    for i in range(nLayers):
        layerOutput = RELU(np.dot(layerOutput, weights[i]) + biases[i])
    return layerOutput

def nn():
    # Initial Parameters
    length = 28
    gridSize = length * length
    interLayerSize = 16
    finalLayerSize = 10
    totalIterations = 1
    maxInit = 0.1

    # Load dataset 
    (data_train, labels_train), (data_test, label_test) = loadMNIST()

    # Initialization of arrays
    W1 = np.random.uniform(-maxInit, maxInit, (gridSize, interLayerSize))
    W2 = np.random.uniform(-maxInit, maxInit, (interLayerSize, interLayerSize))
    W3 = np.random.uniform(-maxInit, maxInit, (interLayerSize, finalLayerSize))
    B1 = np.random.uniform(-maxInit, maxInit, interLayerSize)
    B2 = np.random.uniform(-maxInit, maxInit, interLayerSize)
    B3 = np.random.uniform(-maxInit, maxInit, finalLayerSize)

    # loop for evaluation and learning
    for idx, image in enumerate(data_train):
        output = evaluate(image, 3, [W1, W2, W3], [B1, B2, B3])
        # print result. TODO: Implement backpropagation and learning
        print(f"Index: {idx}, Label: {labels_train[idx]}, Prediction: {np.argmax(output)}" )

if __name__ == '__main__':
    nn()