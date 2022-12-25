import numpy as np
import torch
from torch import nn
from torch import optim

input_dim = 5
hidden_dim = 30
output_dim = 5

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = torch.nn.functional.sigmoid(self.layer_3(x))

        return x

# #def get_nn():
# model = NeuralNetwork(input_dim, hidden_dim, output_dim)
# ## return model

# i = torch.randn(3, 3)
# print(i)


# pred = model.forward(i)


# print(pred)
