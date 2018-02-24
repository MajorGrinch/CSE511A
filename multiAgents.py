# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [
            self.evaluationFunction(gameState, action) for action in legalMoves
        ]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(
            bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates
        ]
        "*** YOUR CODE HERE ***"
        distance = []
        for ghostState in newGhostStates:
            if ghostState.getPosition() == newPos:
                return -99999

        if action == 'Stop':
            return -99999

        foodList = currentGameState.getFood().asList()
        pacmanPos = list(successorGameState.getPacmanPosition())

        def manhattan(x, y):
            return -1 * abs(x[0] - y[0]) - 1 * abs(x[1] - y[1])

        for food in foodList:
            distance.append(manhattan(food, pacmanPos))

        return max(distance)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
     """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

        Directions.STOP:
            The stop direction, which is always legal

        gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def maxvalue(gamestate, depth):
            value = -99999
            if gamestate.isLose() or gamestate.isWin() or depth == 0:
                utility = self.evaluationFunction(gamestate)
                return utility

            State = [gamestate.generateSuccessor(0, x) for x in gamestate.getLegalActions()]
            # for each successor of state:
            for succ in State:
                value = max(value, minvalue(succ, depth, gamestate.getNumAgents() - 1))
            return value

        def minvalue(gamestate, depth, agentIndex):
            # print "agentIndex: ", agentIndex
            val = 99999
            if gamestate.isLose() or gamestate.isWin() or depth == 0:
                utility = self.evaluationFunction(gamestate)
                return utility

            State = [gamestate.generateSuccessor(agentIndex, x) for x in gamestate.getLegalActions(agentIndex)]
            for succ in State:
                if agentIndex == 1:
                    val = min(val, maxvalue(succ, depth - 1))
                elif agentIndex > 1:
                    val = min(val, minvalue(succ, depth, agentIndex - 1))
            return val

        numofghost = gameState.getNumAgents() - 1
        # print "numofghost: ", numofghost
        move = Directions.STOP
        val = -99999
        if gameState.isWin() or gameState.isLose():
            return Directions.STOP
        for action in gameState.getLegalActions():
            if action == Directions.STOP:
                continue
            if val < minvalue(gameState.generateSuccessor(0, action), self.depth, numofghost):
                val = minvalue(gameState.generateSuccessor(0, action), self.depth, numofghost)
                move = action
        print val
        return move


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, depth, alpha, beta):
            if gameState.isLose() or gameState.isWin() or depth == 0:
                return self.evaluationFunction(gameState)
            v = float('-inf')
            successors = [gameState.generateSuccessor(0, x) for x in gameState.getLegalActions()]
            for successor in successors:
                # print "alpha: ", alpha
                v = max(v, minValue(successor, depth, gameState.getNumAgents() - 1, alpha, beta))
                alpha = max(alpha, v)
                if alpha >= beta:
                    # print "prune max"
                    return alpha
            return v

        def minValue(gameState, depth, agentIndex, alpha, beta):
            if gameState.isLose() or gameState.isWin() or depth == 0:
                return self.evaluationFunction(gameState)
            v = float('inf')
            # print "agentIndex: ", agentIndex
            successors = [gameState.generateSuccessor(agentIndex, x) for x in gameState.getLegalActions(agentIndex)]
            for successor in successors:
                if agentIndex == 1:
                    v = min(v, maxValue(successor, depth - 1, alpha, beta))
                else:
                    v = min(v, minValue(successor, depth, agentIndex - 1, alpha, beta))
                beta = min(beta, v)
                if beta <= alpha:
                    # print "prune min"
                    return beta
            return v
        
        num_ghost = gameState.getNumAgents() - 1
        move = Directions.STOP
        val = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        if gameState.isWin() or gameState.isLose():
            return Directions.STOP
        for action in gameState.getLegalActions():
            if action == Directions.STOP:
                continue
            temp = minValue(gameState.generateSuccessor(0, action), self.depth, num_ghost, alpha, beta) 
            if val < temp:
                val = temp
                move = action
            if val >= beta:
                print "prune totally"
                break
            alpha = max(alpha, val)
        # print "val: ", val
        return move


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, depth):
            if gameState.isLose() or gameState.isWin() or depth == 0:
                return self.evaluationFunction(gameState)
            v = float('-inf')
            successors = [gameState.generateSuccessor(0, x) for x in gameState.getLegalActions()]
            for successor in successors:
                v = max(v, expValue(successor, depth, gameState.getNumAgents() - 1))
            return v

        def expValue(gameState, depth, agentIndex):
            if gameState.isLose() or gameState.isWin() or depth == 0:
                return self.evaluationFunction(gameState)        
            v = 0
            successors = [gameState.generateSuccessor(0, x) for x in gameState.getLegalActions()]
            p = 1.0 / len(successors)
            for successor in successors:
                if agentIndex == 1:
                    v += p * maxValue(successor, depth - 1)
                else:
                    v += p * expValue(successor, depth, agentIndex - 1)
            return v

        num_ghost = gameState.getNumAgents() - 1
        # print "numofghost: ", numofghost
        move = Directions.STOP
        val = float('-inf')
        if gameState.isWin() or gameState.isLose():
            return Directions.STOP
        for action in gameState.getLegalActions():
            if action == Directions.STOP:
                continue
            temp = expValue(gameState.generateSuccessor(0, action), self.depth, num_ghost)
            if val < temp:
                val = temp
                move = action
        # print val
        return move


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
        Returns an action.  You can use any method you want and search to any depth you want.
        Just remember that the mini-contest is timed, so you have to trade off speed and computation.

        Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
        just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
