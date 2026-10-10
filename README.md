# Handwritten Digit Recognition (From Scratch)

An experimental feedforward neural network and backpropagation algorithm built **from scratch in pure NumPy**, designed to recognize handwritten digits from the MNIST dataset. It includes an interactive GUI that allows you to draw digits with your mouse and test the network's predictions in real time.

---

## Overview

The core purpose of this project was an experiment in understanding the fundamental mathematics of deep learning. Instead of using high-level frameworks (such as PyTorch or TensorFlow) for model construction and automatic differentiation, the entire neural network pipeline was written by hand:

- **Matrix-based Forward Propagation**: Linear transformations with ReLU and Softmax activations.
- **Analytical Backpropagation**: Gradient derivations calculated and updated across layers using mini-batch gradient descent.
- **Image Preprocessing**: Centering and bounding-box normalization to match MNIST dimensions.
- **Interactive Canvas**: A Tkinter-based canvas to test custom handwritten digits drawn with a mouse.

> *Note: TensorFlow/Keras is used exclusively as a utility to download the MNIST dataset.*

---

## Network Architecture

| Layer | Type | Neurons | Activation |
| :--- | :--- | :--- | :--- |
| **Input** | Grayscale image vector ($28 \times 28$) | 784 | None |
| **Hidden 1** | Fully Connected | 16 | ReLU |
| **Hidden 2** | Fully Connected | 16 | ReLU |
| **Output** | Fully Connected | 10 (digits 0–9) | Softmax |

- **Training Strategy**: Mini-batch Gradient Descent (default batch size: 256, learning rate: 0.05).
- **Initialization**: Uniform distribution $\mathcal{U}(-\text{maxInit}, \text{maxInit})$.

---

## Project Structure

```text
HandwrittenDigitRecognition/
├── neuralNetwork.py      # Core NeuralNetwork class (forward pass, backprop, training loop)
├── main.py               # Dataset loading, training orchestration, and Tkinter GUI
├── requirements.txt      # Project dependencies
├── .gitignore            # Git exclusion rules
├── LICENSE               # MIT License
└── README.md             # Project documentation
```

---

## Getting Started

### 1. Prerequisites

Ensure you have Python 3.8+ installed on your machine.

### 2. Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/pedro-villa/HandwrittenDigitRecognition.git
cd HandwrittenDigitRecognition
pip install -r requirements.txt
```

### 3. Running the Project

To start training and enter the interactive testing mode:

```bash
python main.py
```

1. **Training Phase**: The network will train on the 60,000 MNIST training images for 40 epochs and display accuracy evaluations against the test set.
2. **Interactive Drawing Mode**: A black canvas will appear asking you to draw a specific digit. Draw using your mouse, then close the canvas window to see the network's prediction and running accuracy in your terminal.

---

## License

This project is open-source under the [MIT License](LICENSE).
