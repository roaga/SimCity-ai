import math
import numpy as np
from enum import Enum

time = 0 # game time

funds = 1000 # player's cash
happiness = 100 # happiness of city
population = 0 # population of city
utilities = 0 # total water/energy provided for the city

map_dimensions = (10, 10) # should be 256x256 for proper game
building_map = np.zeros(map_dimensions)
population_map = np.zeros(map_dimensions)
happiness_map = np.zeros(map_dimensions)
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
    UTILITY_PLANT = 10

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
def calculateDistance(x1, x2, y1, y2):
	return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2));

# Checks if a radius is free of buildings / map edges around an area w/ particular center and radius
def checkIfRadiusFree(building_map, centerX, centerY, radius):
	# TODO: Maybe find a better way of iterating over a circular area?
	if (centerX - radius < 0 or centerX + radius > len(building_map)):
		return False;
	if (centerY - radius < 0 or centerY + radius > len(building_map[0])): # Assumes non-jagged array with at least one column
		return False;
	for r in range(centerX - radius, centerX + radius):
		for c in range(centerY - radius, centerY + radius):
			if (calculateDistance(r, c, centerX, centerY) <= radius):
				# Grid coordinate is within radius
				if (building_map[r, c] != 0):
					return False;
	return True;

def placeBuilding(buildingNum, row, col):
    #check if coords are valid (road access, not occupied)
    radiusToCheck = 2; # Not sure what this should be
    radiusFree = checkIfRadiusFree(building_map, row, col, radiusToCheck);

    #check for build cost and purchase
    if (radiusFree):
        print("Placing building...");
		
		# Place the building
    else:
        print("Building cannot be in range of another building.");
        return False;

    #update maps

    print("TODO")
   
def destroyBuilding(row, col):
    #update maps

    print("TODO")

def wait():
    return

def collectTaxes():
    # calculate and add to funds

    print("TODO")

def takeTurn(action):
    # bot chooses between place building, destroy building, and wait

    collectTaxes()
    time += 1

    print(building_map)
    print("TODO")
