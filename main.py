
from hex_engine import hexPosition
import mcts
import numpy as np

import torch.nn.functional as F

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import os

size = 5

input_dim = size
output_dim = size

model_path = "./model.pt";

class NeuralNetwork(nn.Module):
    

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, size)
       
    def forward(self, x): 
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = F.normalize(y)

        for i in range(0,size):
            for j in range(0,size):
                if x[i][j] != 0.0:
                    y[i][j] = 0.0

        return y


if(__name__ == "__main__"):


    model = NeuralNetwork()

    if os.path.exists(model_path):
        file = torch.load(model_path)
        model.load_state_dict(file)
        model.eval()

    learning_rate = 0.1
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_values = []    

    good_guesses = []

    print(model)

    for i_ in range(0,30):

        myboard = hexPosition(size=size)
        while myboard.winner == 0:
            
            t = torch.tensor(myboard.get_float_state())
            pred = model(t)

            ms = mcts.mctsagent(state = myboard)
            ms.search(0.2)        
            best_move = ms.best_move()            

            y = ms.get_tensor_matrix()
            y = torch.tensor(y)

            loss = loss_fn(pred, y)
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()

            myboard.play(best_move)

            myboard.calc_winner()

            if myboard.winner != 0:
                break

            ms = mcts.mctsagent(state = myboard)
            ms.search(0.1)        
            best_move = ms.best_move() 
            myboard.play(best_move)
            myboard.calc_winner()

            
        print(i_)
    

    torch.save(model.state_dict(),model_path)

    plt.plot(np.array(loss_values))
    plt.plot(np.array(good_guesses))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("./sv.png")
    plt.show()