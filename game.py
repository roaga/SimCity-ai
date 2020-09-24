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

# Typical distance calculation function
def calculateDistance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2));

# Checks if a radius is free of buildings / map edges around an area w/ particular center and radius
def checkIfOnGrid(x, y, grid):
    return not (x < 0 or x > len(grid) or y < 0 or y > len(grid[0])); # Assumes non-jagged array with at least one column 

def checkIfRadiusFree(building_map, centerX, centerY, radius):
    # TODO: Maybe find a better way of iterating over a circular area?
    '''
    if (centerX - radius < 0 or centerX + radius + 1 > len(building_map)):
        return False;
    if (centerY - radius < 0 or centerY + radius + 1 > len(building_map[0])): # Assumes non-jagged array with at least one column
        return False;
    '''
    
    if (not checkIfOnGrid(centerX, centerY, building_map)):
        return False;
    
    if (radius > 0):
        for r in range(centerX - radius, centerX + radius + 1):
            for c in range(centerY - radius, centerY + radius + 1):
                if (checkIfOnGrid(centerX, centerY, building_map) and calculateDistance(r, c, centerX, centerY) <= radius):
                    # Grid coordinate is within radius
                    # print("Distance of " + str(calculateDistance(r, c, centerX, centerY)) + " at (" + str(r) + "," + str(c) + ")");
                    if (building_map[r, c] != 0):
                        return False;

    else:
        if (building_map[centerX, centerY] != 0):
            return False;                    
    
    return True;
    
def checkIfNearbyRoads(building_map, centerX, centerY, radius):
    
    if (not checkIfOnGrid(centerX, centerY, building_map)):
        return False;
    
    if (radius > 0):
        for r in range(centerX - radius, centerX + radius + 1):
            for c in range(centerY - radius, centerY + radius + 1):
                if (checkIfOnGrid(centerX, centerY, building_map) and calculateDistance(r, c, centerX, centerY) <= radius):
                    # Grid coordinate is within radius
                    if (building_map[r][c] == 1):
                        return True;

    else:
        print("INVALID INPUT FOR CHECKIFNEARBYROADS()! RADIUS CANNOT BE ZERO FOR USEFUL OUTPUT!");                 
    
    return False;

def placeBuilding(buildingNum, row, col):
    #global variables
    global building_map;
    global funds;
    
    #check if input is reasonable
    if (buildingNum == 0):
        print("INVALID INPUT FOR PLACEBUILDING()! BUILDINGNUM CANNOT BE ZERO FOR USEFUL OUTPUT!");   
    
    #check if coords are valid (road access, not occupied)
    insideGrid = checkIfOnGrid(row, col, building_map);
    buildingRadiusToCheck = 0;
    buildingRadiusFree = False;
    roadRadiusToCheck = 1;
    roadRadiusFree = False
    costPermitting = False

    buildingCost = buildings[buildingNum - 1]["buildCost"]

    #check if building inside grid
    if (insideGrid):
        buildingRadiusFree = checkIfRadiusFree(building_map, row, col, buildingRadiusToCheck);
    else:
        print("Building must be inside the grid.");
        return False;
        
    #check if no buildings in way
    if (buildingRadiusFree):
        roadRadiusFree = buildingNum == 1 or checkIfNearbyRoads(building_map, row, col, roadRadiusToCheck);
        costPermitting = (buildingCost <= funds);
    else:
        print("Building cannot be in range of another building.");
        return False;
        
    #check for nearby roads
    if (roadRadiusFree):
        costPermitting = (buildingCost <= funds);
    else:
        print("Building must have nearby roads.");
        return False;
    
    if (costPermitting):
        print("Placing building...");
		
        pop = buildings[buildingNum - 1]["population"] #TODO: calculate population based on services
        buildingRange = buildings[buildingNum - 1]["range"]
	    # Place the building
        funds -= buildingCost;
        #update maps and variables
        building_map[row][col] = buildingNum;
        population += pop
        population_map[row][col] = pop

        # Specific service update
        # if buildingNum == 1 or 2 or 3 or 4:
            # Do nothing; delete this later
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

        return True;
    else:
        print("Invalid funds.");
        return False;
		

    print("TODO")
  
def destroyBuilding(row, col):
    return False

    print("TODO")

def wait():
    return

def computeHappiness():
    for i in range(0, 10): # iterate through map
        for j in range(0, 10):
            if(population_map(i, j) > 0): # check if building is a housing building
                flag = True # flag to check if services have been met
                if(fire_map(i, j) != 1 or police_map(i, j) != 1 or health_map(i, j) != 1): # check if a core service is there
                    happiness = happiness - 300
                    flag = False
                if(school_map(i, j) != 1 or park_map(i, j) != 1 or leisure_map(i, j) != 1): # check if a leisure service is there
                    happiness = happiness - 15
                    flag = False
                if(flag): # if both services are there at a house, increase happiness
                    happiness = happiness + 75
    happiness_percent = happiness // 100 # calculate happiness percentage value

def collectTaxes():
    computeHappiness()
    tax = population * (happiness / 10000)
    return tax
    # calculate and add to funds

def takeTurn(action):
    # bot chooses between place building, destroy building, and wait
    collectTaxes()
    time += 1

    funds = funds + collectTaxes() # update funds
    print("Happiness: " + happiness_percent + "%") # display happiness
    print(building_map)
    
    choice = randint(0, 2) #1 of 3 choices: wait, destroy, or place
    if (choice == 0): #wait
        wait()
    elif (choice == 1): #delete building at (row, col)
        b = False
        row = randint(0, map_dimensions[0])
        col = randint(0, map_dimensions[1])
        while not b:
            row = randint(0, map_dimensions[0])
            col = randint(0, map_dimensions[1])
            b = destroyBuilding(row, col) #keep randomly generating a coordinate to delete until it is able to be deleted
    else: #place building of type choice at (row, col)
        b = False
        row = randint(0, map_dimensions[0])
        col = randint(0, map_dimensions[1])
        choice = randint(1, 10)
        while not b:
            row = randint(0, map_dimensions[0])
            col = randint(0, map_dimensions[1])
            b = placeBuilding(choice, row, col) #keep randomly generating a coordinate to place a building until possible
