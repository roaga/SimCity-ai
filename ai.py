import game 
import tensorflow as tf
import numpy as np
from random import randint
import torch
from torch import nn
#state will be 10 x 10 x 8 array - 10 x 10 for board dimensions and 8 boards. We might need to add to this last dimension later
#action space: all possible actions on a given turn
#reward: happiness - population dependent
#Number of possible actions - can place 11 buildings anywhere or destroy on 10 x 10 grid (12 x 10 x 10) or wait (1)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv = nn.Conv2d(128, 128, 1, 1)
    

    def forward(self, x):
        res = ResBlock()
        for i in range(19):
            x = res.forward(x)
        x = torch.flatten(x) 
        move = nn.Linear(x, 12*10*10+1)
        move = torch.sigmoid(x) #move to be taken
        val = nn.Linear(x, 1) #current state of the board as a number between -1 and 1
        val = torch.tanh(x)
        return x, val


class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out
