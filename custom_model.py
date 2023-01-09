
import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch

num_channels = 512

class OthelloNNet(nn.Module):
    def __init__(self, size):
        # game params
        self.size = size

        super(OthelloNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)

        self.fc1 = nn.Linear(num_channels*(self.size-4)*(self.size-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, x):
        s = x.view(-1, 1, self.size, self.size)
        s = F.relu(self.bn1(self.conv1(s)))    
        s = F.relu(self.bn2(self.conv2(s)))    
        s = F.relu(self.bn3(self.conv3(s)))    
        s = F.relu(self.bn4(self.conv4(s)))    
        s = s.view(-1, num_channels*(self.size-4)*(self.size-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=0.3, training=True)  
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=0.3, training=True)  

        pi = self.fc3(s)
        pi = pi.masked_fill(x != 0.0, 0.0)
        pi = pi.abs()
        m = pi.sum()
        pi = pi/m         
        v = self.fc4(s)   

        return pi, torch.tanh(v)