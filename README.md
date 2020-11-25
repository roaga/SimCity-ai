# SimCity-ai
An ML project to play a command line version of SimCity.

# Building Types
1 = ROAD, costs $0
2 = HOUSE, costs $1000
3 = TOWER, costs $10,000
4 = SKYSCRAPER, costs $20,000
5 = FIRE_STATION, costs $15,000
6 = POLICE_STATION, costs $15,000
7 = HOSPIITAL, costs $15,000
8 = SCHOOL, costs $15,000
9 = PARK, costs $15,000
10 = LEISURE, costs $15,000
11 = UTILITY_PLANT, costs $5,000

# Introduction
placeBuilding(buildingNum, row, col) => Places a building of a specified type down, replacing whatever was in the space before. Funds changed based on price.
placeBuildingIfPossible(buildingNum, row, col) => Checks to make sure it is possible to place a building down in the specified spot, then calls placeBuilding().
destroyBuilding(row, col) => Replaces an index on grid with a blank one, refunds the price of the building
destroyBuildingIfPossible(row, col) => If possible to destroy a building, calls destroyBuilding()
checkIfNearbyRoads(building_map, centerX, centerY, radius) => Check if there are nearby roads in a certain radius
updateRange(building_map, centerX, centerY, radius) => Update all buildings with a range (on respective grids)
checkIfOnGrid(x, y, grid) => Check if point is on grid
calculateDistance(x1, y1, x2, y2) => Calculate the distance between two points

computeHappiness() => Currently unused

# Monte Carlo Tree Search (MCTS)
The Node class:
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
  

# QuickMap
  - Never used, but is a potentially more memory-efficient way to access large grids
  - Instead of storing empty grid locations, it keeps a dynamic dictionary/"HashMap" of spots that were filled and what they were filled with
The tree is made up of node objects with children. 
