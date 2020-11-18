import math
import random

grid = [0, 0, 0, 0, 1, 0, 0, 0, 0]
#grid = [0, 0, 0, 0, 0]

posX = 4
turns_left = 50

def drawGrid():
    print(grid)


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
            #if (self.children[i].times_checked > 0):
                #print(str(self.children[i].action) + " " + str(self.children[i].value / self.children[i].times_checked) + " " + str(self.children[i].times_checked))
            #else:
                #print(str(self.children[i].action) + " inf")
            if (self.children[i].times_checked > 0 and self.children[i].value / self.children[i].times_checked > bestMoveValue):
                bestMove = i
                bestMoveValue = self.children[i].value / self.children[i].times_checked
        # Move!
        if bestMove != -1:
            keepGoing = simulateAction(self.children[bestMove].action)
            return keepGoing
        else:
            # ERROR!
            print("Error in Monte Carlo Tree Search! See pickBestMove() function!")
            return False



    # Explore the branch of children
    def exploreBranch(self, layer):

        # Simulate if there is no children
        if (len(self.children) == 0):
            #print("There are no child nodes... " + str(self.times_checked))
            # CREATE NEW CHILD NODES TO EXPLORE
            if (self.times_checked != 0 or self.parent == None):
                #print("Creating new child nodes...")
                # create multiple nodes depending on game state
                # move down to a child node
                # run exploreBranch() again from child node, which will simulate the game then backpropagate (code rest of backpropagation too)
                
                randPossTurns = getRandomPossibleTurns(3)
                for i in randPossTurns:
                    self.createChildNode(0, i)

            # SIMULATE A GAME
            else:
                #print("Simulating game...")
                self.value = simulateGame(10)
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
        keepGoing = simulateAction(nodeToExplore.action)
        if (not keepGoing):
            result = 0
        else:
            # Keep on exploring nodes!
            layer += 1
            result = nodeToExplore.exploreBranch(layer)

            # Undo the turns
            if (nodeToExplore.action == 0):
                moveRight()
            else:
                moveLeft()

        self.value += result
        self.times_checked += 1


        
        #print("Result: " + str(self.value))
        return result

def moveLeft():
    global posX
    if (posX - 1 >= 0):
        grid[posX-1] = 1
        grid[posX] = 0
        posX -= 1
        return True
    else:
        return False
    
def moveRight():
    global posX
    if (posX + 1 < len(grid)):
        grid[posX+1] = 1
        grid[posX] = 0
        posX += 1
        return True
    else:
        return False   
    
def getPossibleTurns():
    global num_poss_destroy_moves

    # Return all possible turns
    possible_turns = []
    possible_turns.append(0) # left
    possible_turns.append(1) # right

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
    keepGoing = simulateAction(randTurn)

    # Recursively keep simulating turns
    if (keepGoing == False):
        result = 0
    else:
        if (num_turns == 0):
            result = 1 # win!
        else:
            result = simulateGame(num_turns - 1) # keep simulating games
    


    # Undo the turns
    if (keepGoing):
        if (randTurn == 0):
            moveRight()
        else:
            moveLeft()

    # Return the results
    #print("Simulation funds: " + str(-simulationFunds/100)) 
    return result

def simulateAction(action):
    if (action == 0):
        return moveLeft()
    else:
        return moveRight()

moveLeft()
moveRight()
moveRight()
moveRight()
moveRight()
moveRight()
moveRight()
moveRight()
moveRight()
moveRight()
moveRight()
moveRight()
moveRight()
drawGrid();

while True:
    # Monte Carlo Tree Search test
    parentNode = Node(None, 0, -1)
    for i in range(3000):
        parentNode.exploreBranch(0)
        #if (i % 55 == 0):
            #print("Training: " + str(i) + "/3000")
    keepGoing = parentNode.pickBestMove()
    if (not keepGoing):
        break
    print(grid)
    '''val = input("Continue? Y/N")
    if val == "N":
        break
    else:
        continue'''
