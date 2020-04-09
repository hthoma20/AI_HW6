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
        super(AIPlayer,self).__init__(inputPlayerId, "hw6")
    
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
        moves = listAllLegalMoves(currentState)
        selectedMove = moves[random.randint(0,len(moves) - 1)];

        #don't do a build move if there are already 3+ ants
        numAnts = len(currentState.inventories[currentState.whoseTurn].ants)
        while (selectedMove.moveType == BUILD and numAnts >= 3):
            selectedMove = moves[random.randint(0,len(moves) - 1)];
            
        return selectedMove

    def reward(self, currentState):
        myState = currentState.fastclone()
        me = myState.whoseTurn
        enemy = abs(me - 1)
        myInv = getCurrPlayerInventory(myState)
        myFood = myInv.foodCount
        enemyInv = getEnemyInv(self, myState)
        tunnels = getConstrList(myState, types=(TUNNEL,))
        myTunnel = tunnels[1] if (tunnels[0].coords[1] > 5) else tunnels[0]
        enemyTunnel = tunnels[0] if (myTunnel is tunnels[1]) else tunnels[1]
        hills = getConstrList(myState, types=(ANTHILL,))
        myHill = hills[1] if (hills[0].coords[1] > 5) else hills[0]
        enemyHill = hills[1] if (myHill is hills[0]) else hills[0]
        enemyQueen = enemyInv.getQueen()

        foods = getConstrList(myState, None, (FOOD,))

        myWorkers = getAntList(myState, me, (WORKER,))
        myOffense = getAntList(myState, me, (SOLDIER,))
        enemyWorkers = getAntList(myState, enemy, (WORKER,))

        # "steps" val that will be returned
        occupyWin = 0

        # keeps one offensive ant spawned
        # at all times
        if len(myOffense) < 1:
            occupyWin += 20
        elif len(myOffense) <= 2:
            occupyWin += 30

        # encourage more food gathering
        if myFood < 1:
            occupyWin += 20

        # never want 0 workers
        if len(myWorkers) < 1:
            occupyWin += 100
        if len(myWorkers) > 1:
            occupyWin += 100

        # want to kill enemy queen
        if enemyQueen == None:
            occupyWin -= 1000

        # calculation for soldier going
        # to kill enemyworker and after
        # going to sit on enemy anthill
        dist = 100
        for ant in myOffense:
            if len(enemyWorkers) == 0:
                if not enemyQueen == None:
                    dist = approxDist(ant.coords, enemyHill.coords)
            else:
                dist = approxDist(ant.coords, enemyWorkers[0].coords) + 10
                if len(enemyWorkers) > 1:
                dist += 10

        occupyWin += (dist) + (enemyHill.captureHealth)

        # Gather food
        foodWin = occupyWin
        if not len(myOffense) > 0:
            foodNeeded = 11 - myFood
            for w in myWorkers:
                distanceToTunnel = approxDist(w.coords, myTunnel.coords)
                distanceToHill = approxDist(w.coords, myHill.coords)
                distanceToFood = 9999
                for food in foods:
                    if approxDist(w.coords, food.coords) < distanceToFood:
                        distanceToFood = approxDist(w.coords, food.coords)
                if w.carrying: # if carrying go to hill/tunnel
                    foodWin += min(distanceToHill, distanceToTunnel) - 9.5
                    if w.coords == myHill.coords or w.coords == myTunnel.coords:
                        foodWin += 1.5
                    if not len(myOffense) == 0:
                        foodWin -= 1
                else: # if not carrying go to food
                    if w.coords == foods[0].coords or w.coords == foods[1].coords:
                        foodWin += 1.2
                        break
                    foodWin += distanceToFood/3 - 1.5
                    if not len(myOffense) == 0:
                        foodWin -= 1
            occupyWin += foodWin * (foodNeeded)

        # encourage queen killing or sitting on tunnel
        if not enemyQueen == None:
            if enemyQueen.coords == enemyHill.coords:
                occupyWin += 20

        return 1-math.exp(-0.001*(occupyWin))
    
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
