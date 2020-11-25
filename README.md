# SimCity-ai
An ML project to play a command line version of SimCity.

# Building Types
  - 1 = ROAD, costs $0
  - 2 = HOUSE, costs $1000
  - 3 = TOWER, costs $10,000
  - 4 = SKYSCRAPER, costs $20,000
  - 5 = FIRE_STATION, costs $15,000
  - 6 = POLICE_STATION, costs $15,000
  - 7 = HOSPIITAL, costs $15,000
  - 8 = SCHOOL, costs $15,000
  - 9 = PARK, costs $15,000
  - 10 = LEISURE, costs $15,000
  - 11 = UTILITY_PLANT, costs $5,000

# Introduction
  - placeBuilding(buildingNum, row, col) => Places a building of a specified type down, replacing whatever was in the space before. Funds changed based on price.
  - placeBuildingIfPossible(buildingNum, row, col) => Checks to make sure it is possible to place a building down in the specified spot, then calls placeBuilding().
  - destroyBuilding(row, col) => Replaces an index on grid with a blank one, refunds the price of the building
  - destroyBuildingIfPossible(row, col) => If possible to destroy a building, calls destroyBuilding()
  - checkIfNearbyRoads(building_map, centerX, centerY, radius) => Check if there are nearby roads in a certain radius
  - updateRange(building_map, centerX, centerY, radius) => Update all buildings with a range (on respective grids)
  - checkIfOnGrid(x, y, grid) => Check if point is on grid
  - calculateDistance(x1, y1, x2, y2) => Calculate the distance between two points

  - computeHappiness() => Currently unused


# QuickMap
  - Never used, but is a potentially more memory-efficient way to access large grids
  - Instead of storing empty grid locations, it keeps a dynamic dictionary/"HashMap" of spots that were filled and what they were filled with
The tree is made up of node objects with children. 

# Reinforcement Learning
Reinforcement Learning is a form of artificial intelligence commonly used to play games. In the process, there is an agent, or the actual AI, and environment, the game itself, where the agent is attempting to maximize some reward value given by the environment. During each state of the environment, the agent selects from a set of available actions, which then transitions the environment to the next state and modifies the reward value. The agent's goal is to maximize the reward value at the conclusion of the game by selecting the optimal actions for each state. Traditionally, this has been done using Q learning, where the agent keeps a table of values that map each state to the highest possible reward values for that state. However, for more complex games such as chess and sim city, there are too many states to keep track of with modern computer limitations. 

AlphaZero is a chess program which solves this problem by using a combination of the Monte Carlo Tree Search and a deep neural network. We have implemented this process to train an AI to play our version of SimCity, which can be found in the file game.py.

# Monte Carlo Tree Search (MCTS)
In game.py, The Node class:
  - getNumChildren() => number of children nodes associated with the node
  - addChildNode(node) => Add a pre-existing child node to the parent
  - createChildNode(value) => Create a new child node and add it to the parent
  - calculateExploreVal(node) => Calculate the exploration/exploitation value for the specified node
  - exploreBranch() => Determine which node to explore next, and recursively explore until game should be simulated.

Other functions:
  - simulateAction(action) => Simulates a numeric action
  - getPossibleTurns() => Get the possible turns based on the board state
  - getRandomPossibleTurns() => Randomly select possible turns to use
  - simulateGame() => Get the results of a simulated, randomly-played game
  - turnIsPossible(action) => Determine whether a certain action is possible
  - convertToAction(actionType, num) => Convert a String/integer pair into a single numeric action
  - convert1DtoCoords(action) => Convert a single integer into a coordinate
  - decomposeAction(action) => Transform a numeric action into something more readable
  

# Neural Network
In ai.py, the neural network takes as input the state of the board, which in our case is an array of dimensions 10 x 10 x 8. It outputs two values, the value for the state of the board, and the selected move to be taken at that state. The value is between -1 and 1, where 1 represents a winning state and -1 represents a losing state. The move output is an array of size 1201 for the 1201 possible moves available: 1100 moves for placing 1 of 11 types of buildings at each slot in the 10x10 grid, 100 moves for destroying a building at a spot in the grid, and 1 move for doing nothing that turn. Each value in the 1201 output array is between 0 and 1, and the selected move will be the index with the highest value. The neural network is involves 1 initial convolutional layer, 19 residual layers, and then finally linear layers at the end.

# Training
During training, the Monte Carlo Tree Search will attempt to determine the most optimtal move and corresponding value for that state. The neural network will also calulate values and moves for the state, which will then be compared to the selected values from the MCTS in the loss function. For now, we have decided to use L2 loss (squared error). As such, the idea is that the neural network will learn to approximate values and moves which the MCTS would take for a given state, since it would be too slow to compute the values with the MCTS for every state.
