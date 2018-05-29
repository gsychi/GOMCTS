import torch

class PytorchNN:
    def __init__(self):
        self.device = torch.device('cpu')

        # self.X = X
        # self.y = y

        #self.N, self.input, self.hidden1, self.hidden2, self.hidden3, self.output = 64, len(self.X[0]), 100, 100, 100, len(self.y[0])
        self.N, self.input, self.hidden1, self.hidden2, self.hidden3, self.output = 64, 1000, 100, 100, 100, 10

        self.X = torch.randn(self.N, self.input, device=self.device) #random
        self.y = torch.randn(self.N, self.output, device=self.device)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input, self.hidden1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden1, self.hidden2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden2, self.hidden3),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden3, self.output)
        ).to(self.device)

        self.loss_function = torch.nn.MSELoss(size_average=False)

        self.learning_rate = 0.0004

    def train(self):

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
        return self.model(X)

nn = PytorchNN()
nn.train()