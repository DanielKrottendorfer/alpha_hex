
from hex_engine import hexPosition
import mcts
import numpy as np

import torch.nn.functional as F

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim

size = 3

input_dim = size
hidden_dim = 1250
output_dim = size

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_4 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x) 
        x = self.layer_3(x)
        x = F.relu(self.layer_4(x))
        x = F.normalize(x)
        return x


if(__name__ == "__main__"):


    model = NeuralNetwork()

    learning_rate = 0.1
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_values = []    

    good_guesses = []

    print(model)

    for i_ in range(0,1000):

        myboard = hexPosition(size=size)
        while myboard.winner == 0:
            
            t = torch.tensor(myboard.get_float_state())
            pred = model(t)

            # b_i = (0,0)
            # for i in range(1,size):
            #     for j in range(1,size):
            #         if pred[b_i[0]][b_i[1]] < pred[i][j]:
            #             b_i = [i,j]

            ms = mcts.mctsagent(state = myboard)
            ms.search(0.1)        
            best_move = ms.best_move()            

            # if best_move[0] == b_i[0] & best_move[1] == b_i[1]:
            #     good_guesses.append(1.0)
            # else:   
            #     good_guesses.append(0.0)

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
        
    plt.plot(np.array(loss_values))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()