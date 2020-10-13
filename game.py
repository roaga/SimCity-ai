import math
import numpy as np
from enum import Enum
from random import randint

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

class QuickCoord:
    def __init__(self, val):
        #self.loc = loc
        self.val = val

    
# QUICKMAP is a (possibly?) more space-efficient way of storing grids.
# Methods: .set() and get()
# Variables: sizeX, sizeY to specify grid size, name to help with debugging (optional), defaultVal as the array default value
class QuickMap:
    # Initializes the QuickMap
    def __init__(self, sizeX, sizeY, name="<no name given>", defaultVal=-1):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.coords = {}
        self.name = name;
        self.defaultVal = defaultVal;

    # Converts 2D coord to a dictionary-friendly 1D coord (assumes map dimensions are constant)
    def convertTo1DCoord(self, x, y):
        if ( not (x >= 0 and x < self.sizeX and y >= 0 and y < self.sizeY) ):
            print("Error in QuickMap named " + self.name + "! You're trying to access a coordinate at a location outside of the QuickMap!");
        
        return x + y * self.sizeY;
    
    # Searches for a coord, and returns None if it can't find it
    def searchCoord(self, x, y):
        searchForLoc = self.convertTo1DCoord(x, y);
        return self.coords.get(searchForLoc, None);

    # Adds a coord object
    def insertCoord(self, x, y, val):
        insertLoc = self.convertTo1DCoord(x, y);
        self.coords[insertLoc] = QuickCoord(val);

    # FOR PUBLIC USE: Changes the value of a space or adds a new coord if one does not exist
    def set(self, x, y, val):
        # Search for coord to replace
        coord = self.searchCoord(x, y);

        # Replace val if it exists already
        if (coord != None):
            if (val is self.defaultVal):
                # Remove coord if not needed (default)
                del self.coords[self.convertTo1DCoord(x, y)];
            else:
                coord.val = val;
        # Add val if it does not exist
        else:
            if (val is not self.defaultVal):
                self.insertCoord(x, y, val);
            
    # FOR PUBLIC USE: Gets the value of a specific spot on the grid
    def get(self, x, y):
        result = self.searchCoord(x, y);
        if (result is None):
            result = QuickCoord(self.defaultVal);
        
        return result;

''' 
QUICKMAP FUNCTIONALITY TESTS
quickMap = QuickMap(10, 10, "test", 0);

quickMap.set(1, 1, 1)
quickMap.set(1, 2, -1)
quickMap.set(1, -20, -1)
print(quickMap.get(1, 2).val);
'''

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

def updateRange(mapNum, centerX, centerY, radius, adding):
    if (not checkIfOnGrid(centerX, centerY, building_map) or building_map[r, c] == 0):
        return False
    
    if (radius > 0):
        for r in range(centerX - radius, centerX + radius + 1):
            for c in range(centerY - radius, centerY + radius + 1):
                if (checkIfOnGrid(centerX, centerY, building_map) and checkIfOnGrid(r, c, building_map) and calculateDistance(r, c, centerX, centerY) <= radius):
                    # Grid coordinate is within radius
                    # print("Distance of " + str(calculateDistance(r, c, centerX, centerY)) + " at (" + str(r) + "," + str(c) + ")")
                    # UPDATE BUILDING HERE
                    if adding:
                        if mapNum == 5:
                            fire_map[r, c] += 1
                        elif mapNum == 6:
                            police_map[r, c] += 1
                        elif mapNum == 7:
                            health_map[r, c] += 1
                        elif mapNum == 8:
                            school_map[r, c] += 1
                        elif mapNum == 9:
                            park_map[r, c] += 1
                        elif mapNum == 10:
                            leisure_map[r, c] += 1
                    else: 
                        if mapNum == 5:
                            fire_map[r, c] -= 1
                        elif mapNum == 6:
                            police_map[r, c] -= 1
                        elif mapNum == 7:
                            health_map[r, c] -= 1
                        elif mapNum == 8:
                            school_map[r, c] -= 1
                        elif mapNum == 9:
                            park_map[r, c] -= 1
                        elif mapNum == 10:
                            leisure_map[r, c] -= 1
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
        if buildingNum >= 5 and buildingNum <= 10:
            updateRange(buildingNum, row, col, buildingRange)
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
    popTemp = 0 # value to update population at the end of happiness calculation
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

                 # update happiness for each building
                temp = population_map[i, j]
                population_map[i, j] = (5 * (10 ** (building_map[i, j] - 1))) * happiness_percent
                population = population - (temp - population_map[i, j])
                
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

def getIndex():
    index = 0
    for i in range(10):
        for j in range(10):
            for k in range(8):
                index += getMap(k)[i][j]
    return index

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
            building_num = (action - 1) // 100 - 1
        else:
            building_num = (action) // 100
        return takeTurn((action - 1) // 10, (action - 1) % 10, True, building_num, False)



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
        b = placeBuildingIfPossible(row, col, building_num)
        if (not b):
            return getIndex(), 0
    elif (place == False):
        b = destroyBuildingIfPossible(row, col)
        if (not b):
            return getIndex(), 0
        return getIndex(), population


for i in range(1000):
    takeTurn()
