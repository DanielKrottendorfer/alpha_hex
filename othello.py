
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

channel_num = 512
class OthelloNNet(nn.Module):
    def __init__(self, size):
        # game params
        self.board_x = size
        super(OthelloNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, channel_num, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channel_num, channel_num, 3, stride=1)
        #self.conv4 = nn.Conv2d(channel_num, channel_num, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(channel_num)
        self.bn2 = nn.BatchNorm2d(channel_num)
        self.bn3 = nn.BatchNorm2d(channel_num)
        #self.bn4 = nn.BatchNorm2d(channel_num)

        self.fc1 = nn.Linear(channel_num, 1024)
        #self.fc1 = nn.Linear(channel_num*(self.board_x-4)*(self.board_x-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, channel_num)
        self.fc_bn2 = nn.BatchNorm1d(channel_num)

        self.fc3 = nn.Linear(channel_num, size*size)

        self.training = True

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_x
        s = s.view(-1, 1, self.board_x, self.board_x)                # batch_size x 1 x board_x x board_x
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_x
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_x
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_x-2)
        #s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_x-4)

        #s = s.view(-1, channel_num*(self.board_x-4)*(self.board_x-4))
        s = s.view(-1, channel_num)
        
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))),training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), training=self.training)  # batch_size x channel_num

        pi = self.fc3(s)       
        pi = torch.reshape(pi,(self.board_x,self.board_x))
        return F.softmax(pi)