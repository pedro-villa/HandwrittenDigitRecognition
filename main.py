import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random

def RELU(x): 
    return np.maximum(x, 0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

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
    A = [None] * (nLayers+1)
    Z = [None] * (nLayers)
    A[0] = data
    for i in range(nLayers-1):
        Z[i] = np.dot(A[i], weights[i]) + biases[i]
        A[i+1] = RELU(Z[i])
    last = nLayers-1
    Z[last] = np.dot(A[last], weights[last]) + biases[last]
    A[last+1] = softmax(Z[last])
    return A, Z

def canonical(vector, pos):
    target = np.zeros_like(vector)
    target[pos] = 1
    return target

def cost(vector, label):
    # Calculates square difference between vector and {label}th canonical vector
    return np.sum((vector - canonical(vector, label)) ** 2)

def test(data, labels, nLayers, W, B):
    hits = 0
    totalCases=0
    print("Starting test...")
    for idx, image in enumerate(data):
        output = np.argmax(evaluate(image, nLayers, W, B)[0][-1])
        if output == labels[idx]:
            hits+=1
        totalCases+=1
    print(f"Test ended. {hits}/{totalCases} hits, {100*hits/totalCases}% accuracy.")

def visual_trial_mode(W, B, data_test, label_test):
    """
    Enters interactive test mode that picks random images from test data
    """
    print("\n--- Visual Test Mode ---")
    print("An image from the test set will be shown.")
    print("Close the image window to see the next test case.")
    print("Press Ctrl+C to quit.")

    
    (_, _), (data_test_orig, _) = keras.datasets.mnist.load_data()

    try:
        while True:
            
            idx = random.randint(0, len(data_test) - 1)
            image_vector = data_test[idx]
            image_matrix = data_test_orig[idx]
            true_label = label_test[idx]

            
            A, _ = evaluate(image_vector, 3, W, B)
            prediction = np.argmax(A[-1])

            
            plt.imshow(image_matrix, cmap='gray')
            plt.title(f"Predicción: {prediction} (Etiqueta Real: {true_label})")
            plt.axis('off') 
            plt.show()

    except KeyboardInterrupt:
        print("\nExiting Visual Test Mode.")

def train(data, labels, epochs=10):
    # Initial Parameters
    length = 28
    gridSize = length * length
    interLayerSize = 16
    finalLayerSize = 10
    maxInit = 0.1
    learning_rate = 0.01

    block_size = 100 # Block size for backpropagation loop 

    # Initialization of arrays
    W1 = np.random.uniform(-maxInit, maxInit, (gridSize, interLayerSize))
    W2 = np.random.uniform(-maxInit, maxInit, (interLayerSize, interLayerSize))
    W3 = np.random.uniform(-maxInit, maxInit, (interLayerSize, finalLayerSize))
    B1 = np.random.uniform(-maxInit, maxInit, interLayerSize)
    B2 = np.random.uniform(-maxInit, maxInit, interLayerSize)
    B3 = np.random.uniform(-maxInit, maxInit, finalLayerSize)

    # Epoch Loop
    for epoch in range(epochs): 
        # Reorder data and label arrays
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]   

        # Backpropagation Loop
        for block_idx in range(0, len(data), block_size):
            GW1 = np.zeros_like(W1)
            GW2 = np.zeros_like(W2)
            GW3 = np.zeros_like(W3)
            GB1 = np.zeros_like(B1)
            GB2 = np.zeros_like(B2)
            GB3 = np.zeros_like(B3)

            block_cost = 0
        
            end_of_block = min(block_idx + block_size, len(data))

            for idx in range(block_idx, end_of_block):
                # Forward pass
                A, Z = evaluate(data[idx], 3, [W1, W2, W3], [B1, B2, B3])
                block_cost += cost(A[3], labels[idx])

                # Calculate gradient vector through backpropagation
                error = (A[3] - canonical(A[3], labels[idx]))
                D3 = 2 * error
                D2 = np.dot(D3, W3.T) * (Z[1] > 0).astype(int)
                D1 = np.dot(D2, W2.T) * (Z[0] > 0).astype(int)

                # Update mean gradient variables
                GB1 += D1
                GB2 += D2
                GB3 += D3
                GW1 += np.outer(A[0], D1) 
                GW2 += np.outer(A[1], D2) 
                GW3 += np.outer(A[2], D3) 

            actual_block_size = end_of_block - block_idx
            B1 -= learning_rate * GB1 / actual_block_size
            B2 -= learning_rate * GB2 / actual_block_size
            B3 -= learning_rate * GB3 / actual_block_size
            W1 -= learning_rate * GW1 / actual_block_size
            W2 -= learning_rate * GW2 / actual_block_size
            W3 -= learning_rate * GW3 / actual_block_size

        # Test performance on epoch ending
        print(f"Epoch ending {epoch + 1}. Testing performance...")
        test(data_test, label_test, 3, [W1, W2, W3], [B1, B2, B3])
        
    return [W1, W2, W3], [B1, B2, B3]


if __name__ == '__main__':
    # Load dataset 
    (data_train, labels_train), (data_test, label_test) = loadMNIST()
    W, B = train(data_train, labels_train, 10)
    # test(data_test, label_test, 3, W, B)
    visual_trial_mode(W, B, data_test, label_test)