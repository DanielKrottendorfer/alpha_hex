
import torch
from torch import nn
import torch.nn.functional as F
import hex_engine
import numpy as np
import mcts
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

def translator (string):
    #This function translates human terminal input into the proper array indices.
    number_translated = 27
    letter_translated = 27
    names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(string) > 0:
        letter = string[0]
    if len(string) > 1:
        number1 = string[1]
    if len(string) > 2:
        number2 = string[2]
    for i in range(26):
        if names[i] == letter:
            letter_translated = i
            break
    if len(string) > 2:
        for i in range(10,27):
            if number1 + number2 == "{}".format(i):
                number_translated = i-1
    else:
        for i in range(1,10):
            if number1 == "{}".format(i):
                number_translated = i-1
    return (number_translated, letter_translated)

model =  NeuralNetwork()
if __name__ == '__main__':
    
    model_path = "./model.pt"
    file = torch.load(model_path)
    model.load_state_dict(file)
    model.eval()

    board = hex_engine.hexPosition(size=size)

    machine = True




    while True:
        
        if machine:
            b = np.array(board.get_float_state(),dtype=np.single)
            pred,v = model.forward(torch.tensor(b))
            actions = board.getActionSpace()
            
            max_i = actions[0]
            for j in range(1,len(actions)):
                if pred[max_i[0]][max_i[1]] <= pred[actions[j][0]][actions[j][1]]:
                    max_i = actions[j]
            
            if board.play(max_i):
                break
        else:
            
            ms = mcts.mctsagent(state = board)
            ms.search(roll_outs=800)        
            best_move = ms.best_move()            

            if board.play(best_move):
                break


        board.printBoard()
        

        human_input = translator(input("Enter your moove (e.g. 'A1'): "))

        if board.play(human_input):
            break
    


    board.printBoard()