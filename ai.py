import game 
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
#state will be 10 x 10 x 8 array - 10 x 10 for board dimensions and 8 boards. We might need to add to this last dimension later
#action space: all possible actions on a given turn
#reward: happiness - population dependent
#Number of possible actions - can place 11 buildings anywhere or destroy on 10 x 10 grid (12 x 10 x 10) or wait (1)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.res = ResBlock()

    def forward(self, x):
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        for i in range(19):
            x = self.res.forward(x)
        x = torch.flatten(x) 
        move = nn.Linear(800, 12*10*10+1)(x)
        move = torch.sigmoid(move) #move to be taken
        val = nn.Linear(800, 1)(x)
        val = torch.tanh(val)
        return move, val


class ResBlock(nn.Module):
    def __init__(self, inplanes=8, planes=8, stride=1, downsample=None):
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


def loss(moves, vals, wins, pi):
    #moves will be vector of moves over all training examples
    #vals will be state vals over all training examples
    #wins will be whether the game was won on from that state: -1 or 1
    return np.sum((vals - wins) ** 2 - np.matmul(pi, np.transpose(np.log(moves))), axis=1)
