import math
import numpy as np
from enum import Enum
import random

time = 0 # game time
funds = 1000 + 20000 # player's cash + set larger for testing
happiness = 50 # happiness of city, in units
happiness_percent = happiness // 100 # happiness of city, as a percent, reported to model
population = 0 # population of city
utilities = 0 # total water/energy provided for the city
wait_turns = 0 # turns without doing anything

roadRadiusToCheck = 1

map_dimensions = (10, 10) # should be 256x256 for proper game
building_map = np.zeros(map_dimensions)
population_map = np.zeros(map_dimensions)
fire_map = np.zeros(map_dimensions)
police_map = np.zeros(map_dimensions)
health_map = np.zeros(map_dimensions)
school_map = np.zeros(map_dimensions)
park_map = np.zeros(map_dimensions)
leisure_map = np.zeros(map_dimensions)

num_building_types = 11
num_poss_build_moves = map_dimensions[0] * map_dimensions[1] * num_building_types
num_poss_destroy_moves = map_dimensions[0] * map_dimensions[1]

def getMap(x):
    global building_map
    global population_map
    global fire_map
    global police_map
    global health_map
    global school_map
    global park_map
    global leisure_map

    switcher = {
        0 : building_map,
        1 : population_map,
        2: fire_map,
        3: police_map,
        4: health_map,
        5: school_map,
        6: park_map,
        7: leisure_map
    }
    return switcher[x]

class Node:
    # Initializes a node (for Monte Carlo)
    def __init__(self, parent, value, action):
        self.parent = parent
        self.children = []
        self.value = value
        self.action = action
        
        self.times_checked = 0
            

    def getNumChildren(self):
        # Get the number of children nodes associated with the node
        return len(self.children)

    def addChildNode(self, node):
        # Add a pre-existing child node to the parent
        self.children.append(node)

    def createChildNode(self, value, action):
        # Create a new child node and add it to the parent
        newChildNode = Node(self, value, action)
        self.addChildNode(newChildNode)
    
    def calculateExploreVal(self, node):
        # Calculate the exploratory value for the node
        N = node.parent.times_checked
        n_i = node.times_checked

        # Inf val
        if (n_i == 0):
            return -1

        v_i = node.value / node.times_checked

        # Return the result
        returnVal = v_i + 2 * math.sqrt(math.log(N) / n_i)
        #print("Calculating explore val... " + str(v_i) + " + 2 * math.sqrt(math.log("+str(N)+")/"+str(n_i)+")) = "+str(returnVal))
        return returnVal

    def printTree(self, row):
        if (row == 0):
            print("PARENT NODE: " + str(self.value) + ", " + str(self.times_checked))
            row += 1

        for nodeChild in self.children:
            spaces = ""
            for i in range(row):
                spaces = spaces + "    "
            #print(spaces + "Row #" + str(row) + ": " + str(nodeChild.value) + ", " + str(nodeChild.times_checked))
            nodeChild.printTree(row+1)

    def pickBestMove(self):
        # Pick the best child node and move

        # Pick the highest-valued child node
        bestMove = -1
        bestMoveValue = -99999999999
        
        for i in range(len(self.children)):
            if (self.children[i].times_checked > 0 and self.children[i].value / self.children[i].times_checked > bestMoveValue):
                bestMove = i
                bestMoveValue = self.children[i].value / self.children[i].times_checked
        print(self.children[bestMove].times_checked)
        # Move!
        if bestMove != -1:
            simulateAction(self.children[bestMove].action)
        else:
            # ERROR!
            print("Error in Monte Carlo Tree Search! See pickBestMove() function!")



    # Explore the branch of children
    def exploreBranch(self, layer):
        # Global vars
        global building_map
        global funds

        # Simulate if there is no children
        if (len(self.children) == 0):
            #print("There are no child nodes... " + str(self.times_checked))
            # CREATE NEW CHILD NODES TO EXPLORE
            if (self.times_checked != 0 or self.parent == None):
                #print("Creating new child nodes...")
                # create multiple nodes depending on game state
                # move down to a child node
                # run exploreBranch() again from child node, which will simulate the game then backpropagate (code rest of backpropagation too)
                
                randPossTurns = getRandomPossibleTurns(2000)
                for i in randPossTurns:
                    self.createChildNode(0, i)

            # SIMULATE A GAME
            else:
                #print("Simulating game...")
                self.value = simulateGame(1)
                self.times_checked += 1
                #print("Value of simulated node: " + str(self.value))
                return self.value


        # Find branch to evaluate

        # Variables
        firstNode = self.children[0]
        highestValue = self.calculateExploreVal(firstNode)
        highestBranch = 0

        # Loop over branches
        for i in range(len(self.children)):
            # Get the node
            node = self.children[i]
            #if (node.value != 0):
                #print("Node: " + "," + str(self.value) + "," + str(node.value) + "," + str(layer))

            # Properties of the node
            #print(node.times_checked)
            exploreVal = self.calculateExploreVal(node)
            #print("    Explore val: " + str(exploreVal) + ", Layer: " + str(layer))
            exploreValIsInf = (exploreVal == -1)
            #print(nodeIsInf)

            # Get the highest value and respective branch
            if (exploreValIsInf or highestValue < exploreVal):
                highestValue = exploreVal
                highestBranch = i

                #print("HIGHEST BRANCH: " + str(i) + "," + str(self.children[i].times_checked))
                if (exploreValIsInf):
                    break

        #if (highestValue != 0):
            #print("Highest branch: " + str(highestBranch) + "," + str(len(self.children)) + "," + str(highestValue))

        # Find the node that will be examined
        nodeToExplore = self.children[highestBranch]
        
        # Simulate the node's cooresponding turn
        prevSlotX, prevSlotY, prevSlotBuildingType, prevFunds = simulateAction(nodeToExplore.action)

        # Keep on exploring nodes!
        layer += 1
        result = nodeToExplore.exploreBranch(layer)
        self.value += result
        self.times_checked += 1

        # Undo the turn
        if (prevSlotX != -1 and prevSlotY != -1):
            building_map[prevSlotX][prevSlotY] = prevSlotBuildingType
            funds = prevFunds
        
        #print("Result: " + str(self.value))
        return result

class Building(Enum):
    ROAD = 1
    HOUSE = 2
    TOWER  = 3
    SKYSCRAPER = 4
    FIRE_STATION = 5
    POLICE_STATION = 6
    HOSPITAL = 7
    SCHOOL = 8
    PARK = 9
    LEISURE = 10
    UTILITY_PLANT = 11

buildings = [
    {"name": "Road", "buildCost": 0, "population": 0, "range": 0},
    {"name": "House", "buildCost": 1000, "population": 50, "range": 0},
    {"name": "Tower", "buildCost": 10000, "population": 500, "range": 0},
    {"name": "Skyscraper", "buildCost": 20000, "population": 5000, "range": 0},
    {"name": "Fire Station", "buildCost": 15000, "population": 0, "range": 12},
    {"name": "Police Station", "buildCost": 15000, "population": 0, "range": 10},
    {"name": "Hospital", "buildCost": 15000, "population": 0, "range": 10},
    {"name": "School", "buildCost": 15000, "population": 0, "range": 8},
    {"name": "Park", "buildCost": 15000, "population": 0, "range": 4},
    {"name": "Leisure", "buildCost": 15000, "population": 0, "range": 5},
    {"name": "Utility Plant", "buildCost": 5000, "population": 0, "range": 0}
]

# Typical distance calculation function
def calculateDistance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))

# Checks if a radius is free of buildings / map edges around an area w/ particular center and radius
def checkIfOnGrid(x, y, grid):
    return not (x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0])) # Assumes non-jagged array with at least one column 

def checkIfRadiusFree(building_map, centerX, centerY, radius):
    # TODO: Maybe find a better way of iterating over a circular area?
    '''
    if (centerX - radius < 0 or centerX + radius + 1 > len(building_map)):
        return False
    if (centerY - radius < 0 or centerY + radius + 1 > len(building_map[0])): # Assumes non-jagged array with at least one column
        return False
    '''
    
    if (not checkIfOnGrid(centerX, centerY, building_map)):
        return False
    
    if (radius > 0):
        for r in range(centerX - radius, centerX + radius + 1):
            for c in range(centerY - radius, centerY + radius + 1):
                if (checkIfOnGrid(centerX, centerY, building_map) and checkIfOnGrid(r, c, building_map) and calculateDistance(r, c, centerX, centerY) <= radius):
                    # Grid coordinate is within radius
                    # print("Distance of " + str(calculateDistance(r, c, centerX, centerY)) + " at (" + str(r) + "," + str(c) + ")")
                    if (building_map[r, c] != 0):
                        return False
    else:
        if (building_map[centerX, centerY] != 0):
            return False                    
    
    return True

def updateRange(building_map, centerX, centerY, radius):
    # TODO: I'm copy/pasting this basic function a lot, but idk how to make it better

    if (not checkIfOnGrid(centerX, centerY, building_map)):
        return False
    
    if (radius > 0):
        for r in range(centerX - radius, centerX + radius + 1):
            for c in range(centerY - radius, centerY + radius + 1):
                if (checkIfOnGrid(centerX, centerY, building_map) and checkIfOnGrid(r, c, building_map) and calculateDistance(r, c, centerX, centerY) <= radius):
                    # Grid coordinate is within radius
                    # print("Distance of " + str(calculateDistance(r, c, centerX, centerY)) + " at (" + str(r) + "," + str(c) + ")")
                    if (building_map[r, c] != 0):
                        # TODO: UPDATE BUILDING HERE
                        print("TODO")
    else:
        if (building_map[centerX, centerY] != 0):
            return False                    
    
    return True
    
def checkIfNearbyRoads(building_map, centerX, centerY, radius):
    if (not checkIfOnGrid(centerX, centerY, building_map)):
        return False
    
    if (radius > 0):
        for r in range(centerX - radius, centerX + radius + 1):
            for c in range(centerY - radius, centerY + radius + 1):
                if (checkIfOnGrid(centerX, centerY, building_map) and checkIfOnGrid(r, c, building_map) and calculateDistance(r, c, centerX, centerY) <= radius):
                    # Grid coordinate is within radius
                    if (building_map[r, c] == 1):
                        return True
    else:
        print("INVALID INPUT FOR CHECKIFNEARBYROADS()! RADIUS CANNOT BE ZERO FOR USEFUL OUTPUT!")                 
    
    return False

def destroyBuilding(row, col):
    global building_map
    global funds

    buildingNum = building_map[row][col]
    buildingCost = buildings[int(buildingNum - 1)]["buildCost"]

    funds += buildingCost
    building_map[row][col] = 0 # 0 = nothing
    
def placeBuilding(buildingNum, row, col):
    global building_map
    global funds

    buildingCost = buildings[buildingNum - 1]["buildCost"]

    funds -= buildingCost
    building_map[row][col] = buildingNum

def placeBuildingIfPossible(buildingNum, row, col):
    #global variables
    global building_map
    global population_map
    global fire_map
    global police_map 
    global health_map
    global school_map 
    global park_map
    global leisure_map
    global funds
    global time
    global happiness
    global happiness_percent 
    global population
    global utilities
        
    #check if input is reasonable
    if (buildingNum == 0):
        print("INVALID INPUT FOR PLACEBUILDING()! BUILDINGNUM CANNOT BE ZERO FOR USEFUL OUTPUT!")   
    
    #check if coords are valid (road access, not occupied)
    insideGrid = checkIfOnGrid(row, col, building_map)
    buildingRadiusToCheck = 0
    buildingRadiusFree = False
    roadRadiusToCheck = 1
    roadRadiusFree = False
    costPermitting = False

    buildingCost = buildings[buildingNum - 1]["buildCost"]

    #check if building inside grid
    if (insideGrid):
        buildingRadiusFree = checkIfRadiusFree(building_map, row, col, buildingRadiusToCheck)
    else:
        print("Building must be inside the grid.")
        return False
        
    #check if no buildings in way
    if (buildingRadiusFree):
        roadRadiusFree = buildingNum == 1 or checkIfNearbyRoads(building_map, row, col, roadRadiusToCheck)
        costPermitting = (buildingCost <= funds)
    else:
        print("Building cannot be in range of another building.")
        return False
        
    #check for nearby roads
    if (roadRadiusFree):
        costPermitting = (buildingCost <= funds)
    else:
        print("Building must have nearby roads.")
        return False
    
    if (costPermitting):
        print("Placing building...")
		
        pop = buildings[buildingNum - 1]["population"] #TODO: calculate population based on services
        buildingRange = buildings[buildingNum - 1]["range"]
	    # Place the building
        funds -= buildingCost
        #update maps and variables
        building_map[row, col] = buildingNum
        population += pop
        population_map[row, col] = pop

        # Specific service update
        # if buildingNum == 1 or 2 or 3 or 4:
            # Do nothing delete this later
        if buildingNum == 5:
            updateRange(fire_map, row, col, buildingRange)
        elif buildingNum == 6:
            updateRange(police_map, row, col, buildingRange)
        elif buildingNum == 7:
            updateRange(health_map, row, col, buildingRange)
        elif buildingNum == 8:
            updateRange(school_map, row, col, buildingRange)
        elif buildingNum == 9:
            updateRange(park_map, row, col, buildingRange)
        elif buildingNum == 10:
            updateRange(leisure_map, row, col, buildingRange)
        elif buildingNum == 11:
            utilities += 10

        return True
    else:
        print("Invalid funds.")
        return False
		

    print("TODO")
  
def destroyBuildingIfPossible(row, col):
    global building_map
    global roadRadiusToCheck

    if (not checkIfOnGrid(row, col, building_map)):
        print("Building must be on grid!")
        return False

    if (not building_map[row, col] != 0):
        print("Selected location to destroy is not a building!")
        return False

    # update surrounding buildings
    if (building_map[row, col] == 1): # road!
        updateRange(building_map, row, col, roadRadiusToCheck)

    # destroy the building
    destroyBuilding(row, col)

    return True
    
def wait():
    return

def computeHappiness():
    #global variables
    global happiness
    global happiness_percent 
    global wait_turns

    wait_flag = False # flag to check if the AI is stalling
    for i in range(0, 10): # iterate through map
        for j in range(0, 10):
            if(population_map[i, j] > 0): # check if building is a housing building
                wait_flag = True # update stalling flag
                flag = True # flag to check if services have been met
                wait_turns = 0
                if(fire_map[i, j] != 1 or police_map[i, j] != 1 or health_map[i, j] != 1): # check if a core service is there
                    happiness = happiness - 250
                    flag = False
                if(school_map[i, j] != 1 or park_map[i, j] != 1 or leisure_map[i, j] != 1): # check if a leisure service is there
                    happiness = happiness - 25
                    flag = False
                if(flag): # if both services are there at a house, increase happiness
                    happiness = happiness + 150
    if (not (wait_flag)): # if there were no houses encountered, run through this section
        wait_turns = wait_turns + 1 # increase stall count
        if (wait_turns > 5): # if stall count is too high, start penalizing
            happiness = happiness - 250        
    if (happiness > 10000):
        happiness = 10000
    elif (happiness < 0):
        happiness = 0
    else:
        pass
    happiness_percent = happiness // 100 # calculate happiness percentage value

def collectTaxes():
    computeHappiness()
    tax = (int)(population * (happiness / 10000))
    return tax
    # calculate and add to funds

def reset():
    global time
    global funds
    global happiness
    global happiness_percent
    global population
    global utilities
    global wait_turns
    global roadRadiusToCheck
    global map_dimensions
    global building_map
    global population_map
    global police_map
    global fire_map
    global health_map
    global school_map
    global park_map
    global leisure_map

    time = 0 # game time
    funds = 1000 + 20000 # player's cash + set larger for testing
    happiness = 10000 # happiness of city, in units
    happiness_percent = happiness // 100 # happiness of city, as a percent, reported to model
    population = 0 # population of city
    utilities = 0 # total water/energy provided for the city
    wait_turns = 0 # turns without doing anything

    roadRadiusToCheck = 1

    map_dimensions = (10, 10) # should be 256x256 for proper game
    building_map = np.zeros(map_dimensions)
    population_map = np.zeros(map_dimensions)
    fire_map = np.zeros(map_dimensions)
    police_map = np.zeros(map_dimensions)
    health_map = np.zeros(map_dimensions)
    school_map = np.zeros(map_dimensions)
    park_map = np.zeros(map_dimensions)
    leisure_map = np.zeros(map_dimensions)

def getIndex():
    index = 0
    for i in range(10):
        for j in range(10):
            for k in range(8):
                index += getMap(k)[i][j]
    return index

'''
def takeAction(action):
    #action will be between 0 and 1200 inclusive
    #0 means wait
    if (action == 0):
        return takeTurn(-1, -1, False, -1, True)
    #bewteen 1 and 100 means to destory at pos
    elif (action <= 100):
        return takeTurn((action - 1) // 10, (action - 1) % 10, False, -1, False)
    else:
        #101 to 200 inclusive should correspond to building number 1
        if (action % 100 == 0):
            building_num = (action - 1) // 100
        else:
            building_num = (action) // 100
        return takeTurn((action - 1) // 100, (action - 1) % 10, True, building_num, False)

def getState():
    answer = []
    for i in range(8):
        answer.append(getMap(i))
    return np.asarray(answer)
'''

def decomposeAction(action):
    # Give a more intuitive definition of an action (wait, destroy, and place where?)

    # Globals
    global num_poss_build_moves

    if (action == 0):
        return "wait", 0, -1
    elif (action <= num_poss_build_moves):
        return "build", (action - 1) % (map_dimensions[0] * map_dimensions[1]), math.floor((action - 1) / (map_dimensions[0] * map_dimensions[1]) + 1)
    else:
        return "destroy", action - 1 - num_poss_build_moves, -1

def convert1DToCoords(action):
    global map_dimensions

    return int(action / map_dimensions[0]), int(action % map_dimensions[1])

def convertToAction(actionType, num):
    # Globals
    global num_poss_build_moves

    if (actionType == "wait"):
        return 0
    elif (actionType == "build"):
        return num + 1
    elif (actionType == "destroy"):
        return num + 1 + num_poss_build_moves

def turnIsPossible(action):

    # Globals
    global building_map

    # Determine if a certain action is possible
    actionType, num, buildingType = decomposeAction(action)
    
    if (actionType == "wait"): # it is always possible to wait
        return True
    else:
        # Variables
        xLoc, yLoc = convert1DToCoords(num)

        spaceOccupied = False
		

        # Check if space is occupied
        if (checkIfOnGrid(xLoc, yLoc, building_map) and building_map[xLoc, yLoc] != 0):
            spaceOccupied = True

        # Determine if move is possible or not
        if (actionType == "destroy"):
            return spaceOccupied
        else:
            return not spaceOccupied      
    
def getPossibleTurns():
    global num_poss_destroy_moves

    # Return all possible turns
    poss_moves = 1 + num_poss_build_moves + num_poss_destroy_moves
    possible_turns = []

    possible_turns.append(0) # the computer can always wait

    for action in range(1, poss_moves):
        if (turnIsPossible(action)):
            possible_turns.append(action)

    # Return the results
    return possible_turns        


def getRandomPossibleTurns(num):
    # Get num possible turns for the AI to use
    possibleTurns = getPossibleTurns()

    if (len(possibleTurns) < num):
        return possibleTurns # not enough possible turns (should never happen
                             # in this game due to relationship between building
                             # and destroying)
    else:
        randPossTurns = []

        # Keep transferring nodes to randPossTurns from possibleTurns randomly until full
        while (len(randPossTurns) < num):
            turnToConsider = random.randint(0, len(possibleTurns)-1)
            randPossTurns.append(possibleTurns[turnToConsider])
            possibleTurns.pop(turnToConsider)

        # Return the results
        return randPossTurns

def simulateGame(num_turns):
    # Global vars
    global funds

    # Get a random turn
    randTurn = getRandomPossibleTurns(1)[0]

    # Simulate the turn
    prevSlotX, prevSlotY, prevSlotBuildingType, prevFunds = simulateAction(randTurn)

    # Recursively keep simulating turns
    if (num_turns - 1 > 0):
        simulationFunds = simulateGame(num_turns - 1)
    else:
        # Get the value of the game
        simulationFunds = funds

    # Undo the turns
    if (prevSlotX != -1 and prevSlotY != -1):
        #print("Undo! " + str(prevSlotX) + "," + str(prevSlotY))
        building_map[prevSlotX][prevSlotY] = prevSlotBuildingType
        funds = prevFunds

    # Return the results
    #print("Simulation funds: " + str(-simulationFunds/100)) 
    return simulationFunds/10

def simulateAction(action):
    # Global vars
    global building_map
    global funds

    # Vars to return
    prevSlotX = -1
    prevSlotY = -1
    prevSlotBuildingType = -1
    prevFunds = funds

    # Get the details of the action
    actionType, actionLoc1d, buildingType = decomposeAction(action)
    actionLocX, actionLocY = convert1DToCoords(actionLoc1d)

    # Simulate the action
    if (actionType == "build" or actionType == "destroy"):
        # Get data to return about previous slot
        #print("Building at: " + str(actionLocX) + "," + str(actionLocY))
        prevSlotX = actionLocX
        prevSlotY = actionLocY
        prevSlotBuildingType = building_map[actionLocX][actionLocY]

    # Execute on the action
    if (actionType == "build"):
        placeBuilding(buildingType, actionLocX, actionLocY)
    elif (actionType == "destroy"):
        destroyBuilding(actionLocX, actionLocY)

    # Return information about the previous slot
    return prevSlotX, prevSlotY, prevSlotBuildingType, prevFunds

'''
placeBuildingIfPossible(1, 2, 3)
placeBuildingIfPossible(9, 3, 3)
computeHappiness()
print(happiness)
#print(getRandomPossibleTurns(5))

for i in getRandomPossibleTurns(500):
    print("(" + decomposeAction(i)[0] + "," + str(convert1DToCoords(decomposeAction(i)[1])) + "," + str(decomposeAction(i)[2]) + ")")
'''









'''
def takeTurn(row, col, place, building_num, wait):
    #global variables
    global time
    global funds

    # bot chooses between place building, destroy building, and wait
    time += 1

    funds = funds + collectTaxes() # update funds
    #print("Happiness: " + str(happiness_percent) + "%") # display happiness
    #print(building_map)
    if (wait == True):
        return getIndex(), population
    elif (place == True):
        b = placeBuildingIfPossible(building_num, row, col)
        if (not b):
            return getIndex(), -1
        return getIndex(), population
    elif (place == False):
        b = destroyBuildingIfPossible(row, col)
        if (not b):
            return getIndex(), -1
        return getIndex(), population
'''



while True:
    # Monte Carlo Tree Search test
    parentNode = Node(None, 0, -1)
    for i in range(3000):
        parentNode.exploreBranch(0)
        if (i % 55 == 0):
            print("Training: " + str(i) + "/3000")
    parentNode.pickBestMove()
    #print("Num children: " + str(parentNode.getNumChildren()))
    print(building_map) 
    val = input("Continue? Y/N")
    if val == "N":
        break
    else:
        continue
