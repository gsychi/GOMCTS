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
        self.momentum = 0

        # define weight and bias for input -> hidden and hidden -> input
        self.weight_1 = np.random.uniform(-1.0, 1.0, (len(self.X[0]), self.hiddenNodes))
        self.weight_2 = np.random.uniform(-1.0, 1.0, (self.hiddenNodes, self.hiddenNodes))
        self.weight_3 = np.random.uniform(-1.0, 1.0, (self.hiddenNodes, self.hiddenNodes))
        self.weight_4 = np.random.uniform(-1.0, 1.0, (self.hiddenNodes, self.hiddenNodes))
        self.weight_5 = np.random.uniform(-1.0, 1.0, (self.hiddenNodes, len(self.y[0])))
        self.bias_1 = np.random.random((1, self.hiddenNodes))
        self.bias_2 = np.random.random((1, self.hiddenNodes))
        self.bias_3 = np.random.random((1, self.hiddenNodes))
        self.bias_4 = np.random.random((1, self.hiddenNodes))
        self.bias_5 = np.random.random((1, len(self.y[0])))
        self.lastWeights1 = np.zeros((len(self.X[0]), self.hiddenNodes))
        self.lastWeights2 = np.zeros((self.hiddenNodes, self.hiddenNodes))
        self.lastWeights3 = np.zeros((self.hiddenNodes, self.hiddenNodes))
        self.lastWeights4 = np.zeros((self.hiddenNodes, self.hiddenNodes))
        self.lastWeights5 = np.zeros((self.hiddenNodes, len(self.y[0])))

    def trainNetwork(self, val, alpha):

        for _ in range(val):

            #forward propagation
            a = self.X.dot(self.weight_1)+self.bias_1
            inp = 1 / (1 + np.exp(-a))
            b = inp.dot(self.weight_2)+self.bias_2
            hid = 1 / (1 + np.exp(-b))
            c = hid.dot(self.weight_3)+self.bias_3
            hid2 = 1 / (1 + np.exp(-c))
            d = hid2.dot(self.weight_4)+self.bias_4
            hid3 = 1 / (1 + np.exp(-d))
            e = hid3.dot(self.weight_5)+self.bias_5
            fin = 1 / (1 + np.exp(-e))

            #backward propagation
            fin_delta = (self.y - fin) * (fin * (1 - fin))
            hid3_delta = fin_delta.dot(self.weight_5.T) * (hid3 * (1 - hid3))
            hid2_delta = hid3_delta.dot(self.weight_4.T) * (hid2 * (1 - hid2))
            hid_delta = hid2_delta.dot(self.weight_3.T) * (hid * (1 - hid))
            inp_delta = hid_delta.dot(self.weight_2.T) * (inp * (1 - inp))


            self.weight_5 += ((fin_delta.T.dot(hid3)).T + self.lastWeights5) * alpha
            self.weight_4 += ((hid3_delta.T.dot(hid2)).T + self.lastWeights4) * alpha
            self.weight_3 += ((hid2_delta.T.dot(hid)).T + self.lastWeights3) * alpha
            self.weight_2 += ((hid_delta.T.dot(inp)).T + self.lastWeights2) * alpha
            self.weight_1 += ((inp_delta.T.dot(self.X)).T + self.lastWeights1) * alpha

            self.bias_5 += fin_delta.sum(axis=0) * alpha
            self.bias_4 += hid3_delta.sum(axis=0) * alpha
            self.bias_3 += hid2_delta.sum(axis=0) * alpha
            self.bias_2 += hid_delta.sum(axis=0) * alpha
            self.bias_1 += inp_delta.sum(axis=0) * alpha

            #update last weights
            self.lastWeights5 = (fin_delta.T.dot(hid3)).T * self.momentum
            self.lastWeights4 = (hid3_delta.T.dot(hid2)).T * self.momentum
            self.lastWeights3 = (hid2_delta.T.dot(hid)).T * self.momentum
            self.lastWeights2 = (hid_delta.T.dot(inp)).T * self.momentum
            self.lastWeights1 = (inp_delta.T.dot(self.X)).T * self.momentum

    def predict(self, input):
        updated_inp = 1 / (1 + np.exp(-(np.dot(input, self.weight_1)+self.bias_1)))
        updated_hid = 1 / (1 + np.exp(-(np.dot(updated_inp, self.weight_2)+self.bias_2)))
        updated_hid2 = 1 / (1 + np.exp(-(np.dot(updated_hid, self.weight_3) + self.bias_3)))
        updated_hid3 = 1 / (1 + np.exp(-(np.dot(updated_hid2, self.weight_4) + self.bias_4)))
        updated_fin = 1 / (1 + np.exp(-(np.dot(updated_hid3, self.weight_5)+self.bias_5)))

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