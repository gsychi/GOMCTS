import numpy as np
import random
from sklearn import datasets
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

data = datasets.load_iris()

class NeuralNetwork:

    def __init__(self, X, y, hiddenNodes):

        #number of hidden layer neurons
        self.hiddenNodes = hiddenNodes

        self.X = X
        self.y = y

        # define weight and bias for input -> hidden and hidden -> input
        self.weight_1 = np.random.uniform(-1.0, 1.0, (len(self.X[0]), self.hiddenNodes))
        self.weight_2 = np.random.uniform(-1.0, 1.0, (self.hiddenNodes, len(self.y[0])))
        self.bias_1 = np.random.random((1, self.hiddenNodes))
        self.bias_2 = np.random.random((1, len(self.y[0])))

    def trainNetwork(self):

        for _ in range(10000):

            #forward propagation
            a = self.X.dot(self.weight_1)+self.bias_1
            inp = 1 / (1 + np.exp(-a))
            b = inp.dot(self.weight_2)+self.bias_2
            hid = 1 / (1 + np.exp(-b))

            #backward propagation
            hid_delta = (self.y - hid) * (hid * (1 - hid))
            inp_delta = hid_delta.dot(self.weight_2.T) * (inp * (1 - inp))

            self.weight_2 += (hid_delta.T.dot(inp)).T * 0.02
            self.weight_1 += (inp_delta.T.dot(self.X)).T * 0.02

            self.bias_2 += hid_delta.sum(axis=0) * 0.02
            self.bias_1 += inp_delta.sum(axis=0) * 0.02


    def predict(self, input):
        updated_inp = 1 / (1 + np.exp(-(np.dot(input, self.weight_1)+self.bias_1)))
        updated_hid = 1 / (1 + np.exp(-(np.dot(updated_inp, self.weight_2)+self.bias_2)))

        return updated_hid

# normalize input
X = preprocessing.scale(data.data[:, :4])
# reformat target array to [1,0,0],[0,1,0],[0,0,1] from 0,1,2
y = np.zeros((len(data.target), 3))
for i in range(len(data.target)):
    y[i][data.target[i]] = 1

brain = NeuralNetwork(X, y, 15)
brain.trainNetwork()
index = random.randint(0, len(y))
print(brain.predict(X[index]))
print(y[index])