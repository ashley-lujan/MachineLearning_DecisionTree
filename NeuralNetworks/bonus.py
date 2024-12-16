import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PimaClassifier(nn.Module):
    def __init__(self, x_d, width):
        super().__init__()
        self.hidden1 = nn.Linear(x_d, width, bias=True)
        self.act1 = nn.Tanh()
    ##nn.ReLU(),
        self.hidden2 = nn.Linear(width, width, bias = True)
        self.act2 = nn.Tanh() #todo: change this to threshold
        self.output = nn.Linear(width, 1)
        self.act_output = nn.Sigmoid()

        self._initialize_weights()

    def forward(self, x):
        print(x.shape)
        print("before")
        x = self.act1(self.hidden1(x))
        print('after first')
        x = self.act2(self.hidden2(x))
        print('after second')
        x = self.act_output(self.output(x))
        return x

    def _initialize_weights(self):
      # Apply Xavier initialization to linear layers
      nn.init.xavier_uniform_(self.hidden1.weight)
      nn.init.xavier_uniform_(self.hidden2.weight)
      nn.init.xavier_uniform_(self.output.weight)
      
      # Optional: Initialize biases to zeros
      # nn.init.zeros_(self.hidden1.bias)
      # nn.init.zeros_(self.hidden2.bias)
      # nn.init.zeros_(self.output.bias)

#todo: initalize parameters


if __name__ == '__main__':
    
    dataset = np.loadtxt('datasets/hw5/bank-note/train.csv', delimiter=',')
    print("loaded data")
    d = dataset.shape[1]
    X = dataset[:,0:(d - 1)]
    y = dataset[:,(d-1)]

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    # print('hello wolrd')

    print("Creating model")
    model = PimaClassifier(X.shape[1], 5)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    n_epochs = 100
    batch_size = 1
    #add column of ones to X
    X_ones = torch.ones((X.shape[0], 1))
    X_ = torch.cat((X_ones, X), dim=1)

    print("going to do epoch")
    
    for epoch in range(n_epochs):
        print("in epoch", epoch)
        for i in range(0, len(X_), batch_size):
            Xbatch = X_[i:i+batch_size]
            print(Xbatch.shape)
            y_pred = model(Xbatch)
            ybatch = y[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
