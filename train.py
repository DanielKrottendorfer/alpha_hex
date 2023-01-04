
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

import random as R

import copy

size = 6
channel_num = 512

class NeuralNetwork(nn.Module):
    

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, channel_num, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channel_num, channel_num, 3, stride=1)
        self.conv4 = nn.Conv2d(channel_num, channel_num, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(channel_num)
        self.bn2 = nn.BatchNorm2d(channel_num)
        self.bn3 = nn.BatchNorm2d(channel_num)
        self.bn4 = nn.BatchNorm2d(channel_num)

        self.fc1 = nn.Linear(channel_num*(size-4)*(size-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, size*size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, x):
        #                                                           s: batch_size x board_x x board_y
        s = x.view(-1, 1, size, size)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, channel_num*(size-4)*(size-4))

        s = F.dropout(F.relu(self.fc1(s)), p=0.1, training=True)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc2(s)), p=0.1, training=True)  # batch_size x 512

        pi = self.fc3(s)                                                                  # batch_size x action_size
        v = self.fc4(s)           
        
        pi = pi.view(size,size)       
        pi = pi.masked_fill(x != 0.0, 0.0)
        m = pi.sum()
        pi = pi/m
        return pi, torch.tanh(v)
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

# def do_the_flippedi_flop(board):
        
#     ms = mcts.mctsagent(state = board)
#     ms.search(200)
#     m = ms.get_float_matrix()
#     m = np.array(m / m.sum(),dtype=np.single)
#     b = np.array(board.get_float_state(),dtype=np.single)

#     x = (b,flip_h(b), flip_v(b), flip_v(flip_h(b)))
#     y = (m,flip_h(m), flip_v(m), flip_v(flip_h(m)))

#     return (x,y)

def do_the_flippiti_flip(matrix):
    
    b = matrix
    x = [b,flip_h(b), flip_v(b), flip_v(flip_h(b))]
    return x

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

    training_set = (list(),list(),list())

    for _ in range(0,set_size):

        r_moves = R.randint(0,size)
        r_board = random_board(r_moves)

        x_ = list()
        pi_ = list()
        v_ = list()

        a = 1.0

        while True:

            ms = mcts.mctsagent(state = r_board)
            ms.search(100)
            best_move = ms.best_move()
            
            if r_board.player == 2:
                if r_board.play(best_move):
                    a = -1.0
                    break
                continue

            m = ms.get_float_matrix()
            m = np.array(m / m.sum(),dtype=np.single)
            b = np.array(r_board.get_float_state(),dtype=np.single)

            is_duplicat = False
            for t in training_set[0]:
                if compare_matrix(b,t):
                    is_duplicat = True
                    break
            
            if not(is_duplicat):
                x_.append(b)
                pi_.append(m)
                v_.append(0.0)
            
            if r_board.play(best_move):
                a = 1.0
                break
        
        for i in reversed(range(0,len(x_))):
            v_[i] = a


        for i in range(0,len(x_)):
            training_set[0].extend(do_the_flippiti_flip(x_[i]))
            training_set[1].extend(do_the_flippiti_flip(pi_[i]))
            training_set[2].extend([v_[i],v_[i],v_[i],v_[i]])
    
    return training_set


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
            pred,v = play.forward(torch.tensor(b))
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
                #board.printBoard()
                break
            else:
                bs = board.recodeBlackAsWhite()
                board.board = bs
                board.player = 1
            moves += 1
            ii += 1
    print("wins",m1_wins," ",m2_wins)

    if m1_wins > m2_wins:
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
            pred, v = m.forward(torch.tensor(b))
            actions = board.getActionSpace()
            
            max_i = actions[0]
            for j in range(1,len(actions)):
                if pred[max_i[0]][max_i[1]] <= pred[actions[j][0]][actions[j][1]]:
                    max_i = actions[j]
            
            if board.play(max_i):
                #board.printBoard()
                return 1
        else:
            
            ms = mcts.mctsagent(state = board)
            ms.search(roll_outs=20)        
            best_move = ms.best_move()            

            if board.play(best_move):
                board.printBoard()
                return -1

model =  NeuralNetwork()
model_path = "./model.pt"
loss_values = []
v1 = list()
v = 0

def exit_handler():
    #torch.save(model.state_dict(),model_path)
    
    plt.plot(np.array(loss_values))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("./sv.png")


if __name__ == '__main__':

    #atexit.register(exit_handler)

    if os.path.exists(model_path):
        file = torch.load(model_path)
        model.load_state_dict(file)
        model.eval()
    
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr = 0.01)
    epochs = 0
    while True:

        #model_backup = copy.deepcopy(model)

        (x,y,v_) = gen_trainingset(5)
        for i in range(0,len(x)):
            pred,v = model.forward(torch.tensor(x[i]))
            loss = loss_fn(pred, torch.tensor(y[i]))
            loss_v = loss_fn(v,torch.tensor(v_[i]))

            total_loss = loss + loss_v

            total_loss.backward()
            loss_values.append(loss.item())
            optimizer.step()
            print(loss.item(),'\n', loss_v.item(),'\n')

        print(loss)
        #model = self_play(model,model_backup)
        v += mctsMatch(model)
        v1.append(v)

        epochs += 1
        
        torch.save(model.state_dict(),model_path)
        
        plt.plot(np.array(loss_values))
        plt.title("Step-wise Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig("./sv.png")


