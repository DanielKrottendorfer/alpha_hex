
from hex_engine import hexPosition
import mcts
import numpy as np

import torch.nn.functional as F

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import os
import multiprocessing

import random as R

import copy

size = 5

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
    for i in range(s):
        for j in range(s):
            flipped_board[j][i] = board[s-1-j][i]
    return flipped_board
    
def flip_v (board):
    s = len(board)
    flipped_board =  np.zeros(shape=(size,size),dtype=np.single)
    for i in range(s):
        for j in range(s):
            flipped_board[j][i] = board[j][s-1-i]
    return flipped_board

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

def new_set(pid,return_dict):
    r_board = hexPosition(size=size)

    x_ = list()
    pi_ = list()
    v_ = list()

    a = 1.0

    while True:

        ms = mcts.mctsagent(state = r_board)
        ms.search(800)
        best_move = ms.best_move()
        
        if r_board.player == 2:
            if r_board.play(best_move):
                a = -1.0
                break
            continue

        m = ms.get_float_matrix()
        m = np.array(m / m.sum(),dtype=np.single)
        b = np.array(r_board.get_float_state(),dtype=np.single)

        x_.append(b)
        pi_.append(m)
        v_.append(0.0)
        
        if r_board.play(best_move):
            a = 1.0
            break
    
    for i in reversed(range(0,len(x_))):
        v_[i] = a

    result = list(),list(),list()

    for i in range(0,len(x_)):
        result[0].extend(do_the_flippiti_flip(x_[i]))
        result[1].extend(do_the_flippiti_flip(pi_[i]))
        result[2].extend([v_[i],v_[i],v_[i],v_[i]])
    
    return_dict.append(result)

def gen_trainingset_pool(set_size):
    
    manager = multiprocessing.Manager()
    return_dict = manager.list()
    jobs = []
    for i in range(set_size):
        p = multiprocessing.Process(target=new_set, args=(i,return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    print("Program finished!")
    
    temp = list(),list(),list()
    for r in return_dict:
        temp[0].extend(r[0])
        temp[1].extend(r[1])
        temp[2].extend(r[2])
    return temp


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
            ms.search(800)
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

    for i in range(0,20):

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
            
            #board.printBoard()
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
        
        #board.printBoard()
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
                return -1

import custom_model
model =  custom_model.OthelloNNet(size)
model_path = "./model.pt"
loss_values = []
v1 = list()
v = 0


if __name__ == '__main__':


    if os.path.exists(model_path):
        file = torch.load(model_path)
        model.load_state_dict(file)
    model.eval()

    def loss_v(target, output):
        return torch.sum((target - output) ** 2)

    optimizer = optim.Adam(model.parameters(),lr = 0.1)
    epochs = 0

    loss_mse = torch.nn.MSELoss()

    while True:

        model_backup = copy.deepcopy(model)

        (x,y,v_) = gen_trainingset_pool(6)
        for i in range(0,len(x)):
            pred,v = model.forward(torch.tensor(x[i]))

            lp = loss_mse(pred, torch.tensor(y[i]))
            lv = loss_v(v,torch.tensor(v_[i]))
            total_loss = lv + lp

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_values.append(lp.item())

        model = self_play(model,model_backup)

        print(total_loss)
        v += mctsMatch(model)
        v1.append(v)

        epochs += 1
        
        torch.save(model.state_dict(),model_path)
        
        plt.plot(np.array(loss_values))
        plt.title("Step-wise Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig("./sv.png")


