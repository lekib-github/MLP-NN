import dill as pickle
import numpy as np
import numpy_datasets as ds

# 1 (See README.md for numbered comments...)
data = ds.emnist.load('digits', './')
x_test, y_test = data['test_set/images'], data['test_set/labels']
x_train, y_train = data['train_set/images'], data['train_set/labels']
# Scaling pixel values 0-255 -> 0-1 and reshaping 28x28 'images' (28x28 pixel value matrix) to 1x(28*28)
x_train = (x_train/255).reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = (x_test/255).reshape(len(x_test), np.prod(x_test.shape[1:]))


# Activation function for layer-layer transitions, numpy operations used because faster(?) but more importantly
# n will always be a numpy array (see 'sample' func.) and these functions know how to work with them in a sensible way.
def sigmoid(n: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-n))


# Loss function giving a numerical evaluation of the performance of a network on one sample
def binary_cross_entropy(outputarr: np.ndarray, onehotarr: np.ndarray) -> float:
    loss = 0
    outputarr = np.clip(outputarr, 1e-7, 1-1e-7)  # Bounding values in output layer so no instability with log
    for i in range(len(outputarr)):
        # Binary cross entropy formula, used with sigmoid function on multi-label classification problems
        loss += -(onehotarr[i] * np.log(outputarr[i]) + (1-onehotarr[i]) * np.log(1-outputarr[i]))
    return float(loss)


# 2
class Network:
    def __init__(self, ninputnodes: int, nhiddenlayers: int, nhiddennodes: int, noutputnodes: int):
        self.weights = []  # Initializing weights for all nodes in hidden layers and output layer -1 < w < 1
        for i in range(nhiddenlayers):
            if i == 0:  # Special case for 1. hidden layer, the one following input layer
                # 3
                self.weights.append(((np.random.random((nhiddennodes, ninputnodes))) * 2) - 1)
            else:
                self.weights.append(((np.random.random((nhiddennodes, nhiddennodes))) * 2) - 1)
        self.weights.append(((np.random.random((noutputnodes, nhiddennodes))) * 2) - 1)

        self.biases = []  # Initializing biases for all nodes in hidden layers and output layer -0.01 < b < 0.01
        for i in range(nhiddenlayers):
            self.biases.append((((np.random.random((nhiddennodes, 1))) * 2) - 1) / 100)
        self.biases.append((((np.random.random((noutputnodes, 1))) * 2) - 1) / 100)

        # 4
        self.activations = []
        self.activations.append(np.array([]))
        for i in range(nhiddenlayers):
            self.activations.append(np.array([]))
        self.activations.append(np.array([]))

    def sample(self, samplearr: np.ndarray):
        self.activations[0] = samplearr.reshape(samplearr.shape[0], 1)
        for i in range(1, len(self.activations)):
            # Matrix multiplication between weights and activations summed with biases, forward propagation
            self.activations[i] = sigmoid(np.matmul(self.weights[i-1], self.activations[i-1]) + self.biases[i-1])

    # 5
    def backpropagation(self, onehotarr: np.ndarray, learnrate: float):
        # 6
        delta = self.activations[-1] - onehotarr
        for i in range(len(self.weights)-1, -1, -1):
            if i == len(self.weights)-1:
                # 7
                self.biases[i] -= delta*learnrate
                self.weights[i] -= np.matmul(self.activations[i], delta.T).T*learnrate
            else:
                # 8
                delta = np.matmul(self.weights[i+1].T, delta) * (self.activations[i+1]*(1 - self.activations[i+1]))
                # 9
                self.biases[i] -= delta * learnrate
                self.weights[i] -= np.matmul(self.activations[i], delta.T).T * learnrate

    # 10
    def train(self, iterations: int, learnrate: float):
        for i in range(iterations):
            print(f'{"-"*100}\nIteration {i+1}:')
            avgloss = 0
            for j in range(x_train.shape[0]):
                self.sample(x_train[j])
                #  One-hot encoding the label in a ground truth array
                onehotarr = np.zeros((len(self.activations[-1]), 1))
                onehotarr[y_train[j]] = 1
                avgloss += binary_cross_entropy(self.activations[-1], onehotarr)
                if j != 0 and j % (x_train.shape[0]//10) == 0:
                    avgloss /= x_train.shape[0]/10
                    print(f'Average loss for examples {j-x_train.shape[0]//10}-{j}: {avgloss}')
                    avgloss = 0
                self.backpropagation(onehotarr, learnrate)
            # OK to test on the same test data multiple times, the network doesn't do backpropagation on it so it
            # never 'remembers' it, this is just fun data to be displayed as the network learns.
            print(f'The performance of the network after {i+1} iterations is: {self.test()*100}%')

    # Tests over the given test data at a point in time by running an example and checking label, for all test examples.
    def test(self) -> float:
        count = 0
        for i in range(x_test.shape[0]):
            self.sample(x_test[i])
            # Following statement should be safe, as our loss function doesn't motivate all outputs to be close to 1...
            if np.argmax(self.activations[-1].reshape(self.activations[-1].shape[0])) == y_test[i]:
                count += 1
        return count/x_test.shape[0]
