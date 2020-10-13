import game 
#import tensorflow as tf
import numpy as np
from random import randint

#state will be 10 x 10 x 8 array - 10 x 10 for board dimensions and 8 boards. We might need to add to this last dimension later
#action space: all possible actions on a given turn
#reward: happiness - population dependent
#Number of possible actions - can place 11 buildings anywhere or destroy on 10 x 10 grid (12 x 10 x 10) or wait (1)

Q = np.zeros((10 * 10 * 8, 10 * 10 * 12 + 1))
lr = .8
y = .95
num_episodes = 2000
rList = []
for i in range(num_episodes):
    game.reset()
    s = int(game.getIndex())
    j = 0
    while j < 99:
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        a = int(np.argmax(Q[s,:]))
        a += randint(1,10 * 10 * 12 + 1)*(1./(i+1))
        a %= 10 * 10 * 12 + 1
        a = int(a)
        #Get new state and reward from environment
        #print(1)
        s1,r = game.takeAction(a)
        print(r)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        #rAll += r
        s = s1
    #jList.append(j)
    rList.append(rAll)

j = 0
game.rest()
while j < 99:
    a = a = np.argmax(Q[s,:] + np.random.randn(1,10 * 10 * 12 + 1)*(1./(i+1)))
    game.takeAction(a)
    print(game.building_map)
    print(game.population)
    
