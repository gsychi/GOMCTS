import torch
from torch.autograd import Variable
import numpy as np

class PytorchNN:
    def __init__(self, X, y):
        self.device = torch.device('cpu')

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        print(self.X.dtype)
        print(self.y.dtype)

        self.N, self.input, self.hidden1, self.hidden2, self.hidden3, self.output = int(X.shape[0]), int(len(X[0])), 100, 100, 100, int(len(y[0]))

        print(self.input)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input, self.hidden1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden1, self.output)
            # torch.nn.ReLU(),
            # torch.nn.Linear(self.hidden2, self.hidden3),
            # torch.nn.ReLU(),
            # torch.nn.Linear(self.hidden3, self.output)
        ).to(self.device)

        self.loss_function = torch.nn.MSELoss(size_average=False)

        self.learning_rate = 0.0004

    def train(self):
        print("training begins")
        for t in range(500):
            predicted_y = self.model(self.X)
            loss = self.loss_function(predicted_y, self.y)
            print(t, loss.item())
            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for parameter in self.model.parameters():
                    parameter.data -= parameter.grad * self.learning_rate

    def predict(self, X):
        pred = self.model(torch.from_numpy(X).float()).detach().numpy()
        pred = np.reshape(pred, (1,9))
        return pred
