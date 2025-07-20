import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random
from neuralNetwork import NeuralNetwork
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

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

def center_image(img):
    """
    Centra el dígito dibujado en una imagen de 28x28.
    Esto ayuda a que la imagen se parezca más a las del dataset MNIST.
    """
    try:
        # Encontrar el bounding box del dígito
        coords = np.column_stack(np.where(img > 0))
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        
        # Recortar la imagen al bounding box
        cropped = img[y0:y1+1, x0:x1+1]

        # Crear un lienzo de 28x28 y pegar el dígito en el centro
        side = max(cropped.shape)
        # Añadimos un pequeño padding para que no ocupe todo el borde
        padded_side = int(side * 1.4) 
        centered = Image.new("L", (padded_side, padded_side), 0)
        
        paste_x = (padded_side - cropped.shape[1]) // 2
        paste_y = (padded_side - cropped.shape[0]) // 2
        centered.paste(Image.fromarray(cropped), (paste_x, paste_y))

        # Redimensionar a 28x28
        centered = centered.resize((28, 28), Image.LANCZOS)
        return np.array(centered)

    except ValueError:
        # Si la imagen está vacía, devuelve una imagen en negro
        return np.zeros((28, 28))

def draw_and_predict(nn, number=None):
    WIDTH, HEIGHT = 280, 280  # 10x MNIST scale
    window = tk.Tk()
    if number == None: 
        window.title("Draw a number and close the window to see the network's prediction")
    else:
        window.title(f"Draw the number {number} and close the window to see the network's prediction")

    canvas = tk.Canvas(window, width=WIDTH, height=HEIGHT, bg='black')
    canvas.pack()
    image = Image.new("L", (WIDTH, HEIGHT), 'black')
    draw = ImageDraw.Draw(image)

    last_x, last_y = None, None

    def xy(event):
        nonlocal last_x, last_y
        last_x, last_y = event.x, event.y

    def add_line(event):
        nonlocal last_x, last_y
        x, y = event.x, event.y
        if last_x is not None and last_y is not None:
            canvas.create_line(last_x, last_y, x, y, width=20, fill='white', capstyle=tk.ROUND, smooth=True)
            draw.line([last_x, last_y, x, y], fill='white', width=20)
        last_x, last_y = x, y

    canvas.bind('<Button-1>', xy)
    canvas.bind('<B1-Motion>', add_line)

    window.mainloop()

    # Procesar imagen: centrar, normalizar y aplanar
    img_array = np.array(image)
    
    # Centrar la imagen para que se parezca más a los datos de MNIST
    centered_img_array = center_image(img_array)

    img = centered_img_array.astype('float32') / 255.0
    img = img.reshape(28*28)

    # Predecir
    A, _ = nn.evaluate(img)
    prediction = np.argmax(A[-1])
    print(f"Neural Network prediction: {prediction}")
    return prediction

if __name__ == '__main__':
    # Load dataset 
    (data_train, labels_train), (data_test, labels_test) = loadMNIST()
    inputSize = 28*28
    layerSizes = [inputSize, 16, 16, 10]
    
    # Create neural network
    nn = NeuralNetwork(layerSizes)

    # Train network
    print("Starting training...")
    nn.train(data_train, labels_train, trainingTests=True, blockSize=256, epochs=40, learningRate=0.05, data_test=data_test, labels_test=labels_test)
    print("Training ended.")

    # Test network
    # visual_trial_mode(nn, data_test, labels_test)

    print("Entering manual test mode. Please draw the number requested by subsequent texts")
    hits = 0
    misses = 0
    while True:
        real_label = random.randint(0, 9)
        print(f"\nPlease, draw number: {real_label} (or exit manual test mode pressing Ctrl+c)")
        prediction = draw_and_predict(nn)
        if prediction == real_label:
            hits += 1
            print("Success!")
        else:
            misses += 1
            print(f"Failure! Network predicted {prediction}.")
        percentage = (hits / (hits + misses)) * 100 if hits + misses > 0 else 0
        print(f"Aciertos: {hits}/{hits + misses} ({percentage:.1f}%)")