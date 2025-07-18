import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random
from neuralNetwork import NeuralNetwork

def loadMNIST():
    (data_train, labels_train), (data_test, label_test) = keras.datasets.mnist.load_data()
    # Normalize data
    data_train = data_train.astype('float32')/255
    data_test = data_test.astype('float32')/255
    # Reshape array
    data_train = data_train.reshape(-1, 28*28)
    data_test = data_test.reshape(-1, 28*28)

    return (data_train, labels_train), (data_test, label_test)

def visual_trial_mode(nn : NeuralNetwork, data_test, label_test):
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

            
            A, _ = nn.evaluate(image_vector)
            prediction = np.argmax(A[-1])

            
            plt.imshow(image_matrix, cmap='gray')
            plt.title(f"Predicción: {prediction} (Etiqueta Real: {true_label})")
            plt.axis('off') 
            plt.show()

    except KeyboardInterrupt:
        print("\nExiting Visual Test Mode.")

if __name__ == '__main__':
    # Load dataset 
    (data_train, labels_train), (data_test, labels_test) = loadMNIST()
    inputSize = 28*28
    layerSizes = [inputSize, 16, 16, 10]
    
    # Create neural network
    nn = NeuralNetwork(layerSizes)

    # Train network
    print("Starting training...")
    nn.train(data_train, labels_train, trainingTests=True, epochs=20, learningRate=0.1, data_test=data_test, labels_test=labels_test)
    print("Training ended.")

    # Test network
    visual_trial_mode(nn, data_test, labels_test)
