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
        self.weight_2 = np.random.uniform(-1.0, 1.0, (self.hiddenNodes, self.hiddenNodes))
        self.weight_3 = np.random.uniform(-1.0, 1.0, (self.hiddenNodes, len(self.y[0])))
        self.bias_1 = np.random.random((1, self.hiddenNodes))
        self.bias_2 = np.random.random((1, self.hiddenNodes))
        self.bias_3 = np.random.random((1, len(self.y[0])))

    def trainNetwork(self, val, alpha):

        for _ in range(val):

            #forward propagation
            a = self.X.dot(self.weight_1)+self.bias_1
            inp = 1 / (1 + np.exp(-a))
            b = inp.dot(self.weight_2)+self.bias_2
            hid = 1 / (1 + np.exp(-b))
            c = hid.dot(self.weight_3)+self.bias_3
            fin = 1 / (1 + np.exp(-c))

            #backward propagation
            fin_delta = (self.y - fin) * (fin * (1 - fin))
            hid_delta = fin_delta.dot(self.weight_3.T) * (hid * (1 - hid))
            inp_delta = hid_delta.dot(self.weight_2.T) * (inp * (1 - inp))

            self.weight_3 += (fin_delta.T.dot(hid)).T * alpha
            self.weight_2 += (hid_delta.T.dot(inp)).T * alpha
            self.weight_1 += (inp_delta.T.dot(self.X)).T * alpha

            self.bias_3 += fin_delta.sum(axis=0) * alpha
            self.bias_2 += hid_delta.sum(axis=0) * alpha
            self.bias_1 += inp_delta.sum(axis=0) * alpha


    def predict(self, input):
        updated_inp = 1 / (1 + np.exp(-(np.dot(input, self.weight_1)+self.bias_1)))
        updated_hid = 1 / (1 + np.exp(-(np.dot(updated_inp, self.weight_2)+self.bias_2)))
        updated_fin = 1 / (1 + np.exp(-(np.dot(updated_hid, self.weight_3)+self.bias_3)))

        return updated_fin

# normalize input
X = preprocessing.scale(data.data[:, :4])
# reformat target array to [1,0,0],[0,1,0],[0,0,1] from 0,1,2
y = np.zeros((len(data.target), 3))
for i in range(len(data.target)):
    y[i][data.target[i]] = 1

"""
brain = NeuralNetwork(X, y, 15)
brain.trainNetwork(10000)
index = random.randint(0, len(y))
print(brain.predict(X[index]))
print(y[index])
"""