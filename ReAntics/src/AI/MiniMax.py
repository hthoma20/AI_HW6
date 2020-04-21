#Authors: Harry Thoma and Qi Liang

import math
import random
import sys
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *

MY_ID= 0
ENEMY_ID= 1

##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "MiniMax")
    
    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        global MY_ID, ENEMY_ID
        MY_ID= currentState.whoseTurn
        ENEMY_ID= 1 - MY_ID

        #random.seed(373298298)
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    
    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):
        #chache the fastest route between food and tunnel/anthill
        buildCache(currentState)

        abRoot = ABSearchNode(None, currentState, None)
        abRoot.expand(maxDepth=3)

        return abRoot.bestChild.move
    
    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    #registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon):
        #method templaste, not implemented
        pass

#Class to represent a MiniMax node and subtree with alpha-beta pruning
class ABSearchNode:

    #given a parent SearchNode and a Move, create a new SearchNode
    def __init__(self, move, state, parent):
        self.parent = parent
        self.move = move
        self.state = state

        if self.parent == None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

        # we are a max node if it is our turn
        self.maxNode = state.whoseTurn == MY_ID

        self.initAlphaBeta()

        #assess utility, we need this for all nodes (even non-terminal) because we
        #expand in sorted order
        self.evaluation = utility(state, MY_ID) - utility(state, ENEMY_ID)

        #track whose turn it is
        if move is None:
            self.turn = 0
        elif move.moveType == END:
            self.turn = parent.turn + 1
        else:
            self.turn = parent.turn

        # stop if good enough
        if self.depth == 0:
            self.beta = self.evaluation + 20

        self.children = []

    #initilize alpha-beta values by correctly iheriting from ancestors
    def initAlphaBeta(self):
        self.alpha = -math.inf
        self.beta = math.inf

        seenMax = False
        seenMin = False

        #find first matching ancestor
        currAncestor = self.parent

        while currAncestor is not None and not seenMax and not seenMin:
            if currAncestor.maxNode and not seenMax:
                self.alpha = currAncestor.alpha
            elif not currAncestor.maxNode and not seenMin:
                self.beta = currAncestor.beta

            currAncestor = currAncestor.parent

    # is this an interesting node?
    # i.e should we expand it
    def interesting(self, maxDepth):
        # root node is interesting
        if self.move is None:
            return True

        #dont expand past a win node
        if getWinner(self.state) is not None:
            return False

        if self.depth >= maxDepth:
            return False

        #stop expanding when it's the enemy's turn
        if self.parent.turn == 1:
            return False

        return True

    # recursively expand the subtree rooted at this node, up to maxDepth deep
    def expand(self, maxDepth):
        # is this a terminal node?
        if not self.interesting(maxDepth):
            self.alpha = self.evaluation
            self.beta = self.evaluation
            return

        allMoves = listAllLegalMoves(self.state)

        # create and sort children based on utility evaluation
        self.children = [ABSearchNode(move, getNextStateAdversarial(self.state,move), self) for move in allMoves]
        self.children.sort(reverse=True)
        for child in self.children:

            child.expand(maxDepth)

            if self.maxNode:
                self.alpha = max(self.alpha, child.alpha)
            else:
                self.beta = min(self.beta, child.beta)

            # stop expanding if no longer relevant
            if self.alpha >= self.beta:
                break

        if self.maxNode:
            self.bestChild = max(self.children)
        else:
            self.bestChild = min(self.children)

        self.evaluation = self.bestChild.evaluation

    def __str__(self):
        return '<{} {} {}>'.format(self.move, self.evaluation, self.depth)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return self.evaluation < other.evaluation

# hold non-changing but relevant values for the utility function
# in particular, the fastest route between food and tunnel/anthill (deposit)
class Cache:
    def __init__(self, state):
        self.foodCoords = [0]*2
        self.depositCoords = [0]*2
        self.rtt = [0]*2

        foods = getConstrList(state, None, (FOOD,))
        for player in [0,1]:
            deposits = getConstrList(state, player, (ANTHILL, TUNNEL))

            #find the best combo, based on steps to reach one to the other
            bestCombo = min([(d, f) for d in deposits for f in foods], key=lambda pair: stepsToReach(state, pair[0].coords, pair[1].coords))

            self.depositCoords[player] = bestCombo[0].coords
            self.foodCoords[player] = bestCombo[1].coords

            self.rtt[player] = approxDist(self.depositCoords[player], self.foodCoords[player])+1

globalCache = None

def buildCache(state):
    global globalCache

    if globalCache is None or not cacheValid(state):
        globalCache = Cache(state)

#check whether the cache still refers to the current game
def cacheValid(state):
    allFood = [food.coords for food in getConstrList(state, None, (FOOD,))]
    allDeposits = [deposit.coords for deposit in getConstrList(state, None, (ANTHILL, TUNNEL))]
    return all(foodCoord in allFood for foodCoord in globalCache.foodCoords) and \
           all(depositCoord in allDeposits for depositCoord in globalCache.depositCoords)

# evaluate the utility of a state from a given player's perspective
# return a tuple of relevant unweighted components
def utilityComponents(state, perspective):
    enemy = 1-perspective

    # get lists for ants
    myWorkers = getAntList(state, perspective, types=(WORKER,))
    enemyWorkers = getAntList(state, enemy, types=(WORKER,))

    myWarriors = getAntList(state, perspective, types=(DRONE,SOLDIER,R_SOLDIER))
    enemyWarriors = getAntList(state, enemy, types=(DRONE,SOLDIER,R_SOLDIER))

    myQueen = state.inventories[perspective].getQueen()
    enemyQueen = state.inventories[enemy].getQueen()

    foodCoords = globalCache.foodCoords[perspective]
    depositCoords = globalCache.depositCoords[perspective]
    anthillCoords = state.inventories[perspective].getAnthill().coords

    # it's bad if the queen is on the food
    queenInTheWayScore = 0

    queenCoords = myQueen.coords
    if queenCoords in [foodCoords, depositCoords, anthillCoords]:
        queenInTheWayScore -= 1

    queenHealthScore = myQueen.health

    workerDistScore = 0
    workerDangerScore = 0
    for worker in myWorkers:

        # If the worker is carrying food, add the distance to the tunnel to the score
        if worker.carrying == True:
            distanceFromTunnel = approxDist(worker.coords, depositCoords)
            workerDistScore -= distanceFromTunnel

        # if the worker is not carrying food, add the distance from the food and tunnel to the score
        else:
            distTunnelFood = approxDist(foodCoords, depositCoords)
            workerDistScore -= distTunnelFood
            distanceFromFood = approxDist(worker.coords, foodCoords)
            workerDistScore -= distanceFromFood

        #its bad to be close to enemy warriors
        for warrior in enemyWarriors:
            #warriorRange = UNIT_STATS[warrior.type][RANGE] + UNIT_STATS[warrior.type][MOVEMENT]
            if approxDist(worker.coords, warrior.coords) < 2:
                workerDangerScore -= 1

    # Aim to attack workers, if there are no workers, aim to attack queen
    if len(enemyWorkers) != 0:
        targetCoords = enemyWorkers[0].coords
    else:
        targetCoords = enemyQueen.coords

    warriorDistScore = 0
    # Add distance from fighter ants to their targets to score, with a preference to move vertically
    for warrior in myWarriors:
        warriorDistScore -= (warrior.coords[0] - targetCoords[0])**2
        warriorDistScore -= (warrior.coords[1] - targetCoords[1])**2

    #do we have an attacker?
    attackScore = UNIT_STATS[myWarriors[0].type][ATTACK] if len(myWarriors) == 1 else 0

    # punishment for if the enemy has workers
    enemyWorkerScore = - (len(enemyWorkers) * len(myWarriors))

    # Heavy punishment for not having workers, since workers are needed to win
    noWorkerScore = -1 if len(myWorkers) == 0 else 0

    foodScore = state.inventories[perspective].foodCount

    antCountScore = -len(getAntList(state, MY_ID)) if perspective == MY_ID else 0

    return (queenInTheWayScore, workerDistScore, workerDangerScore, warriorDistScore, enemyWorkerScore,
            noWorkerScore, foodScore, attackScore, antCountScore, queenHealthScore)

# evaluate the given state from the given player's perspective
# by assigning weights to components
def utility(state, perspective):
    INF = 10e6
    # return an arbitrarily small score if we're the winner
    winner = getWinner(state)
    if winner is not None:
        if (winner==1 and state.whoseTurn == perspective) or \
                (winner==0 and state.whoseTurn != perspective):
            return INF
        else:
            return 0


    components = utilityComponents(state, perspective)
    #weights determined emperically, food score weighted by round trip time between food and deposit
    weights = (50, 50, 15, 4, 0,
               300, 100*globalCache.rtt[state.whoseTurn], 250*globalCache.rtt[state.whoseTurn], 300, 10)

    return sum(a*b for a,b in zip(weights, components))

##
# getNextStateAdversarial
#
# we copied this because we wanted to model attacking
#
# Description: This is the same as getNextState (above) except that it properly
# updates the hasMoved property on ants and the END move is processed correctly.
#
# Parameters:
#   currentState - A clone of the current state (GameState)
#   move - The move that the agent would take (Move)
#
# Return: A clone of what the state would look like if the move was made
##
def getNextStateAdversarial(currentState, move):
    # variables I will need
    nextState = getNextState(currentState, move)
    myInv = getCurrPlayerInventory(nextState)
    myAnts = myInv.ants

    # If an ant is moved update their coordinates and has moved
    if move.moveType == MOVE_ANT:
        endingCoord = move.coordList[-1]
        for ant in myAnts:
            if ant.coords == endingCoord:
                ant.hasMoved = True

            #attack an enemy if they are next to the soldier
            if ant.type in [SOLDIER, DRONE, R_SOLDIER]:
                for enemy in nextState.inventories[1-currentState.whoseTurn].ants:
                    if approxDist(enemy.coords, ant.coords) < 2:
                        enemy.health-= UNIT_STATS[ant.type][ATTACK]
                        if enemy.health < 1:
                            nextState.inventories[1 - currentState.whoseTurn].ants.remove(enemy)
                        break

    elif move.moveType == END:
        for ant in myAnts:
            ant.hasMoved = False
        nextState.whoseTurn = 1 - currentState.whoseTurn

    elif move.moveType == BUILD:
        getAntAt(nextState, move.coordList[0]).hasMoved= True

    return nextState