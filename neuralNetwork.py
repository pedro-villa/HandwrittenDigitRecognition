import numpy as np
class NeuralNetwork:
    def __init__(self, layerSize, maxInit=0.1):
        # layerSize is an array of size the number of layers, counting the input layer as the first
        # self.nLayers counts the layers of actual "neurons", which does not include the input layer
        # maxinit is the upper bound to random values generated for parameters during the first iteration 
        self.nLayers = len(layerSize)-1
        self.w = [np.random.uniform(-maxInit, maxInit, (layerSize[idx], layerSize[idx+1])) for idx in range(self.nLayers)]
        self.b = [np.random.uniform(-maxInit, maxInit, layerSize[idx+1]) for idx in range(self.nLayers)]
    
    @staticmethod
    def RELU(x): 
        return np.maximum(x, 0)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    @staticmethod
    def canonical(vector, pos):
        target = np.zeros_like(vector)
        target[pos] = 1
        return target

    def cost(self, vector, label):
        # Calculates square difference between vector and {label}th canonical vector
        return np.sum((vector - self.canonical(vector, label)) ** 2)
    
    def evaluate(self, data):
        nLayers = self.nLayers
        w = self.w
        b = self.b
        A = [None] * (nLayers+1)
        Z = [None] * (nLayers)
        A[0] = data
        for i in range(nLayers-1):
            Z[i] = np.dot(A[i], w[i]) + b[i]
            A[i+1] = self.RELU(Z[i])
        last = nLayers-1
        Z[last] = np.dot(A[last], w[last]) + b[last]
        A[last+1] = self.softmax(Z[last])
        return A, Z
    
    def test(self, data, labels):
        hits = 0
        totalCases = 0
        print("Starting test...")
        for idx, image in enumerate(data):
            guess = np.argmax(self.evaluate(image)[0][-1])
            if guess == labels[idx]:
                hits+=1    
            totalCases+=1
        print(f"Test ended. {hits}/{totalCases} hits, {100*hits/totalCases}% accuracy.")

    def train(self, data_train, labels_train, epochs=10, blockSize=100, learningRate=0.01, trainingTests=False, data_test=None, labels_test=None):
        # Epoch loop
        for epoch in range(epochs):
            # Reorder data and label arrays
            indices = np.random.permutation(len(data_train))
            data_train = data_train[indices]
            labels_train = labels_train[indices]

            # Backpropagation loop
            for blockIdx in range(0, len(data_train), blockSize):
                Gw = [np.zeros_like(W) for W in self.w]
                Gb = [np.zeros_like(B) for B in self.b]

                block_cost = 0
                endOfBlock = min(blockIdx + blockSize, len(data_train))

                for idx in range(blockIdx, endOfBlock):
                    # Forward pass
                    A, Z = self.evaluate(data_train[idx])
                    block_cost += self.cost(A[self.nLayers], labels_train[idx])

                    # Calculate gradient vector through backpropagation
                    error = A[self.nLayers] - self.canonical(A[self.nLayers], labels_train[idx])
                    D = [None] * self.nLayers
                    D[self.nLayers - 1] = 2 * error
                    for i in range(self.nLayers-2, -1, -1):
                        D[i] = np.dot(D[i+1], self.w[i+1].T) * (Z[i]>0).astype(int)                     

                    # Upgrade mean gradient variables
                    Gw = [gw + np.outer(a, d) for gw, a, d in zip(Gw, A[:-1], D)]
                    Gb = [gb + d for gb, d in zip(Gb, D)]

                actualBlockSize = endOfBlock - blockIdx
                self.w = [w - learningRate * gw / actualBlockSize for w, gw in zip(self.w, Gw)]
                self.b = [b - learningRate * gb / actualBlockSize for b, gb in zip(self.b, Gb)]

            # Test performance on epoch ending
            if trainingTests == True:
                print(f"Epoch {epoch+1} ending. Testing performance...")
                self.test(data_test, labels_test)

                