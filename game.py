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
    {"name": "Road", "buildCost": 0, "population": 0, "happiness": 0, "fire": False, "police": False, "health": False, "school": False, "park": False},
    {"name": "House", "buildCost": 0, "population": 0, "happiness": 0, "fire": False, "police": False, "health": False, "school": False, "park": False},
    {"name": "Tower", "buildCost": 0, "population": 0, "happiness": 0, "fire": False, "police": False, "health": False, "school": False, "park": False},
    {"name": "Skyscraper", "buildCost": 0, "population": 0, "happiness": 0, "fire": False, "police": False, "health": False, "school": False, "park": False},
    {"name": "Fire Station", "buildCost": 0, "population": 0, "happiness": 0, "fire": False, "police": False, "health": False, "school": False, "park": False},
    {"name": "Police Station", "buildCost": 0, "population": 0, "happiness": 0, "fire": False, "police": False, "health": False, "school": False, "park": False},
    {"name": "Hospital", "buildCost": 0, "population": 0, "happiness": 0, "fire": False, "police": False, "health": False, "school": False, "park": False},
    {"name": "School", "buildCost": 0, "population": 0, "happiness": 0, "fire": False, "police": False, "health": False, "school": False, "park": False},
    {"name": "Park", "buildCost": 0, "population": 0, "happiness": 0, "fire": False, "police": False, "health": False, "school": False, "park": False},
    {"name": "Leisure", "buildCost": 0, "population": 0, "happiness": 0, "fire": False, "police": False, "health": False, "school": False, "park": False},
    {"name": "Utility Plant", "buildCost": 0, "population": 0, "happiness": 0, "fire": False, "police": False, "health": False, "school": False, "park": False}
]

def placeBuilding(buildingNum, row, col):
    #check if coords are valid (road access, not occupied)

    #check for build cost and purchase

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