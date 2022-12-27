
from hex_engine import hexPosition
import mcts
import numpy as np

import torch.nn.functional as F

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import os

import atexit
import torch.nn.functional as F

size = 4

input_dim = size
output_dim = size

model_path = "./model.pt";


class NeuralNetwork(nn.Module):
    

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.fl1 = nn.Linear(3,3)
        self.conv1 = nn.Conv2d(1,125,kernel_size=(3,3),padding=1,bias=False)
        self.conv2 = nn.Conv2d(125,80,kernel_size=(1,1),padding=0,bias=False)
        self.conv3 = nn.Conv2d(80,1,kernel_size=(1,1),padding=0)
        self.fl2 = nn.Linear(3,3)

       
    def forward(self, x): 

        # y = F.relu(self.conv1(y))
        # y = y.view(y.shape[1])
        # y = F.relu(self.fc1(y))
        # y = torch.reshape(y,(3,3))
        # print(y)

        y = self.fl1(x)
        y = y.view(1,1,y.shape[0],y.shape[1])
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        
        y = y.view(y.shape[2],y.shape[3])

        y = F.relu(self.fl2(y))


        y = F.normalize(y)
        return y

import othello

model = othello.OthelloNNet(size=size)
loss_values = []

def exit_handler():
    torch.save(model.state_dict(),model_path)
    
    plt.plot(np.array(loss_values))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("./sv.png")

atexit.register(exit_handler)

if(__name__ == "__main__"):


    if os.path.exists(model_path):
        file = torch.load(model_path)
        model.load_state_dict(file)
        model.eval()

    learning_rate = 0.2
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr = learning_rate)


    print(model)


    for i in range(0,10000):

        myboard = hexPosition(size=size)
        running_loss = 0.0
        while myboard.winner == 0:
            
            t = torch.tensor(myboard.get_float_state())
            pred = model.forward(t)

            ms = mcts.mctsagent(state = myboard)
            ms.search(roll_outs=10000)        
            best_move = ms.best_move()            

            y = torch.tensor(ms.get_tensor_matrix())
            y = F.normalize(y)

            print(pred)
            print(y)
            print()

            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            myboard.play(best_move)

            myboard.calc_winner()

            if myboard.winner != 0:
                break

            
            myboard.playRandom()
            myboard.calc_winner()
            
            running_loss += loss

        
        loss_values.append(running_loss.item())
        print(running_loss.item())
        if i % 10:
            torch.save(model.state_dict(),model_path)

            plt.plot(np.array(loss_values))
            plt.title("Step-wise Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.savefig("./sv.png")
            #plt.show()