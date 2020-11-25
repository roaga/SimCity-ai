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
        self.initConv = nn.Conv2d(8, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.initBatch = nn.BatchNorm2d(128)
        self.res = ResBlock()

    def forward(self, x):
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        x = F.relu(self.initBatch(self.initConv(x)))
        for i in range(19):
            x = self.res.forward(x)
        x = torch.flatten(x) 
        move = nn.Linear(12800, 12*10*10+1)(x)
        move = torch.sigmoid(move) #move to be taken
        val = nn.Linear(12800, 1)(x)
        val = torch.tanh(val)
        return move, val


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


def loss(vals, wins, moves, picked_moves):
    #moves will be vector of moves over all training examples
    #vals will be state vals over all training examples
    #wins will be whether the game was won on from that state: -1 or 1
    vals = np.asarray(vals)
    wins = np.asarray(wins)
    moves = np.asarray(moves)
    picked_moves = np.asarray(picked_moves)
    vals = vals.astype(np.float32)
    wins = wins.astype(np.float32)
    moves = moves.astype(np.float32)
    picked_moves = picked_moves.astype(np.float32)
    vals = torch.from_numpy(vals)
    wins = torch.from_numpy(wins)
    moves = torch.from_numpy(moves)
    picked_moves = torch.from_numpy(picked_moves)
    wins.requires_grad = True
    vals.requires_grad = True
    moves.requires_grad = True
    picked_moves.requires_grad = True
    return torch.sum((vals - wins) ** 2) + torch.sum((picked_moves - moves)**2)

game.reset()
model = NeuralNet()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


for i in range(15):
    print("Training " + str(i))
    vals = []
    wins = []
    moves = []
    picked_moves = []
    l = 0
    for j in range(5):
        parentNode = game.Node(None, 0, -1)
        for k in range(3000):
            parentNode.exploreBranch(0)
            if (k % 55 == 0):
                print("Training Monte Carlo: " + str(k) + "/3000")
        parentNode.pickBestMove()
        state = parentNode.states[0]
        win = parentNode.vals[0]
        picked_move_temp = parentNode.moves[0]
        picked_move = [0] * 1201
        picked_move[picked_move_temp] = 1
        picked_move = np.asarray(picked_move)
        picked_move = picked_move.astype(np.float32)
        picked_move = torch.from_numpy(picked_move)
        win = np.asarray(win)
        win = win.astype(np.float32)
        win = torch.from_numpy(win)
        state = parentNode.states[0]
        move, val = model.forward(state)
        #move = move.detach().numpy()
        #vals.append(val)
        #moves.append(move)
        #wins.append(win)
        #picked_moves.append(picked_move)
        l += (torch.sum((val - win) ** 2) + torch.sum((picked_move - move)**2))
    optimizer.zero_grad()
    print(l)
    l.backward()
    optimizer.step()


'''PATH = './neural_net.pth'
model.load_state_dict(torch.load(PATH))'''

game.reset()
for _ in range(100):
    move, val = model.forward(game.getState())
    move = torch.argmax(move)
    game.simulateAction(move)
    print(game.building_map)
