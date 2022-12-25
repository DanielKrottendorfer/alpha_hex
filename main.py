
from hex_engine import hexPosition

import model
import mcts
import numpy as np

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim

size = 3

input_dim = size
hidden_dim = 30
output_dim = size

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))

        return x


if(__name__ == "__main__"):


    model = NeuralNetwork(input_dim, hidden_dim, output_dim)

    learning_rate = 0.1
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_values = []    

    for i_ in range(0,100):

        myboard = hexPosition(size=size)
        while myboard.winner == 0:
            
            t = torch.tensor(myboard.get_float_state())
            pred = model(t)

            # b_i = np.ndarray([0,0])
            # biggest = pred[0][0]
            # for i in range(1,size):
            #     for y in range(1,size):
            #         if pred[i][y] > biggest:
            #             biggest = pred[i][y]
            #             b_i = (i,y)



            ms = mcts.mctsagent(state = myboard)
            ms.search(0.1)        
            best_move = ms.best_move()            

            y = np.zeros(shape=(size,size),dtype=np.single)
            y[best_move[0]][best_move[1]] = 1.0

            y = torch.tensor(y)

            loss = loss_fn(pred, y)
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()

            myboard.play(best_move)

            myboard.calc_winner()

            if myboard.winner != 0:
                break

            myboard.playRandom()
            
        print(i_)
        
    plt.plot(np.array(loss_values))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()