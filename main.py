import numpy as np

def getData(gridSize):
    pass

def RELU(x): 
    return np.maximum(x, 0)

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

    # Initialization of arrays
    W1 = np.random.uniform(-maxInit, maxInit, (gridSize, interLayerSize))
    W2 = np.random.uniform(-maxInit, maxInit, (interLayerSize, interLayerSize))
    W3 = np.random.uniform(-maxInit, maxInit, (interLayerSize, finalLayerSize))
    B1 = np.random.uniform(-maxInit, maxInit, interLayerSize)
    B2 = np.random.uniform(-maxInit, maxInit, interLayerSize)
    B3 = np.random.uniform(-maxInit, maxInit, finalLayerSize)
    x = np.random.uniform(0, 1, gridSize) #TODO get real data

    # loop for evaluation and learning
    for _ in range(totalIterations):
        output = evaluate(x, 3, [W1, W2, W3], [B1, B2, B3])
    print(output)

if __name__ == '__main__':
    nn()