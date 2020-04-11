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
import random
import pickle


ALPHA = 0.1
GAMMA = 0.9



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
        
        self.state_utility = pickle.load(open("../dict_dump.txt", "rb"))

        self.probability = -1
        #self.state_utility = {}
        self.previous_state = None
        self.move_count = 0  


    def if_exploring(self):
        
        rand_val = random.random()

        if rand_val <= self.probability:
            return True 

        return False


    def get_utility(self, state):
        if stateCategory(state) not in self.state_utility:
            self.state_utility[stateCategory(state)] = self.get_reward(state)
        return self.state_utility[stateCategory(state)]

    def get_reward(self, state):
        getWin_val = getWinner(state)
        if getWin_val:
            print("Won")
            return 1 
        if getWin_val == 0:
            print("Loss")
            return -1
        if getWin_val == None:
            print("..")
            return -0.01

    def set_utility(self, state, value):
        self.state_utility[stateCategory(state)] = value


    def update_utility(self, currentState):
        if not self.previous_state == None:
            
            updated_utility = self.get_utility(self.previous_state) + ALPHA*(self.get_reward(self.previous_state) + GAMMA*self.get_utility(currentState) - self.get_utility(self.previous_state))
            
            self.set_utility(self.previous_state, updated_utility)

        self.previous_state = currentState


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
        #cloneTest(currentState)
        
        if self.move_count > 10000:
            return None
        self.move_count += 1    


        self.update_utility(currentState)

        moves = listAllLegalMoves(currentState)
        
        move_state_list = []

        for move in moves:
            move_state_list.append((getNextState(currentState, move), move))

        if self.if_exploring():
            return random.choice(moves)
        else:
            return max(move_state_list, key = lambda item: self.get_utility(item[0]))[1]    
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
    #Win
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon):
        file_ptr = open("./dict_dump.txt", "wb")
        pickle.dump(self.state_utility, file_ptr)
        file_ptr.close()
        #method templaste, not implemented
        


# return an object that represents the category of the given ant
def antCategory(ant):
    return (ant.UniqueID, ant.coords)


# return an object that represents the category of the given state
def stateCategory(state):
    return tuple((antCategory(ant) for inventory in state.inventories for ant in inventory.ants))

'''
def cloneTest(state):
    d = {}

    clone1 = state.fastclone()
    clone2 = state.fastclone()

    d[stateCategory(clone1)] = 1
    d[stateCategory(clone2)] = 2

    print(d)
'''
