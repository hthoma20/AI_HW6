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


ALPHA = .6
GAMMA = 0.8



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

        self.use_saved_weights = True

        if self.use_saved_weights:
            self.state_action_utility = pickle.load(open("../dict_dump.txt", "rb"))
        else:
            self.state_action_utility = {}

        self.explore_probability = .7

        self.previous_state = None
        self.previous_move = None
        self.move_count = 0

        self.current_game_states = []


    def if_exploring(self):
        
        rand_val = random.random()

        if rand_val <= self.explore_probability:
            return True 

        return False


    def get_utility(self, state_action):
        state, action = state_action
        if stateCategory(state) not in self.state_action_utility:
            self.state_action_utility[stateCategory(state)] = self.get_reward(state)
        return self.state_action_utility[stateCategory(state)]

    def get_reward(self, state):
        getWin_val = getWinner(state)
        if getWin_val:
            return 1
        if getWin_val == 0:
            return -1

        if getWin_val == None:
            food_reward = getCurrPlayerInventory(state).foodCount * .01
            return -0.01 + food_reward

    def set_utility(self, state_action, value):
        state, action = state_action
        self.state_action_utility[(stateCategory(state), action)] = value

    #update the utility of the given "current_state", given the next_state
    #if there is a winner this state, no current state is needed
    def update_utility(self, current_state_action, next_state_action, won=None):

        if not current_state_action == None:

            if won is None:
                next_utility = self.get_utility(next_state_action)
            else:
                next_utility = 1 if won else -1

            curr_utility = self.get_utility(current_state_action)
            curr_reward = self.get_reward(current_state_action[0])
            
            updated_utility = curr_utility + ALPHA*(curr_reward + GAMMA*next_utility - curr_utility)
            
            self.set_utility(current_state_action, updated_utility)

    #udpate the utility of all states seen this game
    def update_game_utilities(self, hasWon):

        #update the utility of the state right before the end of game
        self.update_utility(self.current_game_states[-1], None, hasWon)

        #update the utilities of the rest of the states
        for i in range(len(self.current_game_states)-2, 0, -1):
            current_state = self.current_game_states[i]
            next_state = self.current_game_states[i+1]
            self.update_utility(current_state, next_state)


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
        if self.move_count > 10000:
            return None
        self.move_count += 1

        buildCache(currentState)

        #self.update_utility(self.previous_state, currentState)
        #self.previous_state = currentState

        moves = listAllLegalMoves(currentState)

        if len(getAntList(currentState, currentState.whoseTurn)) > 3:
            moves = [move for move in moves if move.moveType != BUILD]
        
        move_state_list = []

        for move in moves:
            move_state_list.append((getNextState(currentState, move), move))


        if self.if_exploring():
            move_to_make = random.choice(moves)
        else:
            #select the move that takes you to the state with the highest utility
            move_to_make = max(move_state_list, key = self.get_utility)[1]

        self.current_game_states.append((currentState, self.previous_move))
        self.previous_move = move_to_make
        return move_to_make
    
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


    def registerWin(self, hasWon):
        self.move_count = 0

        #self.update_utility(currentState=None, won=hasWon)
        self.update_game_utilities(hasWon)

        #clear the current game states
        self.current_game_states = []

        file_ptr = open("./dict_dump.txt", "wb")
        pickle.dump(self.state_action_utility, file_ptr)
        file_ptr.close()
        


# return an object that represents the category of the given state
def stateCategory(state):

    return utilityComponents(state, state.whoseTurn)

    workers = getAntList(state, state.whoseTurn, (WORKER,))

    if len(workers) == 0:
        return (None, None, None)

    food_coords = getCurrPlayerFood(None, state)[0].coords
    tunnel_coords = getConstrList(state, state.whoseTurn, (TUNNEL,))[0].coords

    worker = workers[0]

    if(worker.carrying):
        east_steps = tunnel_coords[0] - worker.coords[0]
        north_steps = tunnel_coords[1] - worker.coords[1]
    else:
        east_steps = food_coords[0] - worker.coords[0]
        north_steps = food_coords[1] - worker.coords[1]

    return (worker.carrying, north_steps, east_steps)

    return tuple(ant.coords for ant in state.inventories[state.whoseTurn].ants)

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

    return (queenInTheWayScore, workerDistScore, workerDangerScore, warriorDistScore, enemyWorkerScore,
            noWorkerScore, foodScore, attackScore, queenHealthScore)

'''
def cloneTest(state):
    d = {}

    clone1 = state.fastclone()
    clone2 = state.fastclone()

    d[stateCategory(clone1)] = 1
    d[stateCategory(clone2)] = 2

    print(d)
'''
