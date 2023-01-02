
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

import random as R

import copy

import my_model

size = 5


class NeuralNetwork(nn.Module):
    

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.fl1 = nn.Linear(size,size)
        self.conv1 = nn.Conv2d(1,125,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(125,125,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(125,1,kernel_size=3,padding=1)

        
        self.bn1 = nn.BatchNorm2d(125)
        self.bn2 = nn.BatchNorm2d(125)

        self.fl2 = nn.Linear(size,size)

       
    def forward(self, x): 

        y = self.fl1(x)
        y = y.view(1,1,y.shape[0],y.shape[1])
        y = F.relu(self.bn1(self.conv1(y)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.conv3(y))

        y = y.view(y.shape[2],y.shape[3])
        y = F.softmax(self.fl2(y),dim = 0)
        m = torch.sum(y)
        y = y/m
        
        return y

def print_list(board):
    for line in board:
        print(line)
    
    print()

def compare_matrix(a,b):
    s = len(a)
    for i in range(s):
        for j in range(s):
            if a[i][j] != b[i][j]:
                return False
    return True

def flip_h (board):
    s = len(board)
    flipped_board =  np.zeros(shape=(size,size),dtype=np.single)
    #flipping and color change
    for i in range(s):
        for j in range(s):
            flipped_board[j][i] = board[s-1-j][i]
    return flipped_board
    
def flip_v (board):
    s = len(board)
    flipped_board =  np.zeros(shape=(size,size),dtype=np.single)
    #flipping and color change
    for i in range(s):
        for j in range(s):
            flipped_board[j][i] = board[j][s-1-i]
    return flipped_board

def do_the_flippedi_flop(board):
        
    ms = mcts.mctsagent(state = board)
    ms.search(100)
    m = ms.get_float_matrix()
    m = np.array(m / m.sum(),dtype=np.single)
    b = np.array(board.get_float_state(),dtype=np.single)

    x = (b,flip_h(b), flip_v(b), flip_v(flip_h(b)))
    y = (m,flip_h(m), flip_v(m), flip_v(flip_h(m)))

    return (x,y)

def random_board(rand_moves):

    while True:
        myboard = hexPosition(size=size)

        for _ in range(0,rand_moves*2):
            if myboard.playRandom():
                break
        
        if myboard.winner != 0:
            continue
        else:
            return myboard


def gen_trainingset(set_size):

    training_set = list()

    for _ in range(0,set_size):

        r_moves = R.randint(0,size*2)
        r_board = random_board(r_moves)

        training_sub_set = do_the_flippedi_flop(r_board)

        is_duplicat = False
        for t in training_set:
            if compare_matrix(training_sub_set[0][0],t[0][0]):
                is_duplicat = True
                break
        
        if not(is_duplicat):
            training_set.append(training_sub_set)


    x = list()
    y = list()
    for t in training_set:
        for i in range(0,len(t[0])):
            x.append(t[0][i])
            y.append(t[1][i])
    
    return (x,y)



def self_play(m1,m2):

    m1_wins = 0
    m2_wins = 0

    for i in range(0,2):

        play = m1 if i%2 > 0 else m2
        board = hexPosition(size=size)
        ii = i
        moves = 0
        while True:
            play = m1 if ii%2 > 0 else m2
            
            b = np.array(board.get_float_state(),dtype=np.single)
            pred = play.forward(torch.tensor(b))
            actions = board.getActionSpace()
            
            max_i = actions[0]
            for j in range(1,len(actions)):
                if pred[max_i[0]][max_i[1]] <= pred[actions[j][0]][actions[j][1]]:
                    max_i = actions[j]
            
            if board.play(max_i):
                if ii%2:
                    m1_wins += 1
                else:
                    m2_wins += 1
                break
            else:
                bs = board.recodeBlackAsWhite()
                board.board = bs
                board.player = 1
            moves += 1
            ii += 1
    print("wins",m1_wins," ",m2_wins)

    if m1_wins >= m2_wins:
        return m1
    else:
        return m2

def randomMatch(m):
        
    board = hexPosition(size=size)

    for i in range(0,size*size):
        if i%2 == 0:
            
            b = np.array(board.get_float_state(),dtype=np.single)
            pred = m.forward(torch.tensor(b))
            actions = board.getActionSpace()
            
            max_i = actions[0]
            for j in range(1,len(actions)):
                if pred[max_i[0]][max_i[1]] <= pred[actions[j][0]][actions[j][1]]:
                    max_i = actions[j]
            
            if board.play(max_i):
                #board.printBoard()
                return 1
        else:
            if board.playRandom():
                #board.printBoard()
                return -1


def mctsMatch(m):
        
    board = hexPosition(size=size)

    for i in range(0,size*size):
        if i%2 == 0:
            
            b = np.array(board.get_float_state(),dtype=np.single)
            pred = m.forward(torch.tensor(b))
            actions = board.getActionSpace()
            
            max_i = actions[0]
            for j in range(1,len(actions)):
                if pred[max_i[0]][max_i[1]] <= pred[actions[j][0]][actions[j][1]]:
                    max_i = actions[j]
            
            if board.play(max_i):
                board.printBoard()
                return 1
        else:
            
            ms = mcts.mctsagent(state = board)
            ms.search(roll_outs=20)        
            best_move = ms.best_move()            

            if board.play(best_move):
                board.printBoard()
                return -1

import othello
model =  othello.OthelloNNet(size=size)
model_path = "./model.pt"
loss_values = []
v1 = list()
v = 0

def exit_handler():
    torch.save(model.state_dict(),model_path)
    
    plt.plot(np.array(loss_values))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("./sv.png")


if __name__ == '__main__':

    atexit.register(exit_handler)

    if os.path.exists(model_path):
        file = torch.load(model_path)
        model.load_state_dict(file)
        model.eval()
    
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr = 0.01)
    epochs = 0
    while True:

        model_backup = copy.deepcopy(model)

        (x,y) = gen_trainingset(30)
        for i in range(0,len(x)):
            pred = model.forward(torch.tensor(x[i]))
            loss = loss_fn(pred, torch.tensor(y[i]))
            loss.backward()
            loss_values.append(loss.item())
            optimizer.step()
            #print(loss.item())

        print(loss)
        model = self_play(model,model_backup)
        v += mctsMatch(model)
        v1.append(v)

        epochs += 1

