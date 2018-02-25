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

            State = [
                gamestate.generateSuccessor(0, x)
                for x in gamestate.getLegalActions()
            ]
            # for each successor of state:
            for succ in State:
                value = max(value,
                            minvalue(succ, depth,
                                     gamestate.getNumAgents() - 1))
            return value

        def minvalue(gamestate, depth, agentIndex):
            # print "agentIndex: ", agentIndex
            val = 99999
            if gamestate.isLose() or gamestate.isWin() or depth == 0:
                utility = self.evaluationFunction(gamestate)
                return utility

            State = [
                gamestate.generateSuccessor(agentIndex, x)
                for x in gamestate.getLegalActions(agentIndex)
            ]
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
            if val < minvalue(
                    gameState.generateSuccessor(0, action), self.depth,
                    numofghost):
                val = minvalue(
                    gameState.generateSuccessor(0, action), self.depth,
                    numofghost)
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
            successors = [
                gameState.generateSuccessor(0, x)
                for x in gameState.getLegalActions()
            ]
            for successor in successors:
                # print "alpha: ", alpha
                v = max(v,
                        minValue(successor, depth,
                                 gameState.getNumAgents() - 1, alpha, beta))
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
            successors = [
                gameState.generateSuccessor(agentIndex, x)
                for x in gameState.getLegalActions(agentIndex)
            ]
            for successor in successors:
                if agentIndex == 1:
                    v = min(v, maxValue(successor, depth - 1, alpha, beta))
                else:
                    v = min(v,
                            minValue(successor, depth, agentIndex - 1, alpha,
                                     beta))
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
            temp = minValue(
                gameState.generateSuccessor(0, action), self.depth, num_ghost,
                alpha, beta)
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
        num_ghost = gameState.getNumAgents() - 1
        # print "numofghost: ", numofghost
        move = Directions.STOP
        val = float('-inf')
        if gameState.isWin() or gameState.isLose():
            return Directions.STOP
        for action in gameState.getLegalActions():
            if action == Directions.STOP:
                continue
            temp = self.expMax(
                gameState.generateSuccessor(0, action), 1, self.depth)
            if val < temp:
                val = temp
                move = action
        # print val
        return move

    def expMax(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        num_agents = gameState.getNumAgents()
        if agentIndex == 0:
            v = float('-inf')
            for action in gameState.getLegalActions():
                if action == Directions.STOP:
                    continue
                v = max(v,
                        self.expMax(
                            gameState.generateSuccessor(agentIndex, action),
                            (agentIndex + 1) % num_agents, depth - 1))
        else:
            v = 0
            actions = gameState.getLegalActions(agentIndex)
            p = 1.0 / len(actions)
            for action in actions:
                if action == Directions.STOP:
                    continue
                v += self.expMax(
                    gameState.generateSuccessor(agentIndex, action),
                    (agentIndex + 1) % num_agents, depth) * p
        return v


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    """
    I would use six parameters for my evaluation, namely
    - current score
    - distance to closest food
    - distance to closest normal ghost whose scaredTimer is 0
    - distance to closest scared ghost
    - num of food left
    - num of capsule left
    """
    pacman_pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsuleList = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()

    num_food = len(foodList)
    num_capsule = len(capsuleList)
    scaredGhosts = []
    nonScaredGhosts = []
    closest_food_dist = float('inf')
    closest_capsule_dist = float('inf')

    def getManhattanDistanceList(targets):
        """
        The map function would apply a function to every item of a iterable object
        and return the list of results
        """
        return map(
            lambda item: util.manhattanDistance(pacman_pos, item.getPosition()),
            targets)

    for ghostState in ghostStates:
        if ghostState.scaredTimer == 0:
            nonScaredGhosts.append(ghostState)
        else:
            nonScaredGhosts.append(ghostState)

    for food in foodList:
        closest_food_dist = min(closest_food_dist,
                                manhattanDistance(food, pacman_pos))
    for capsule in capsuleList:
        closest_capsule_dist = min(closest_capsule_dist,
                                   manhattanDistance(capsule, pacman_pos))
    if closest_food_dist == float('inf'):
        closest_food_dist = 0
    if closest_capsule_dist == float('inf'):
        closest_capsule_dist = 0
    if len(scaredGhosts) != 0:
        closest_scared_ghost_dist = min(getManhattanDistanceList(scaredGhosts))
    else:
        closest_scared_ghost_dist = 0
    if len(nonScaredGhosts) != 0:
        closest_normal_ghost_dist = max(0.1, min(getManhattanDistanceList(nonScaredGhosts)))
    else:
        closest_normal_ghost_dist = float('inf')
    """
    The coefficient for each parameter is different, which means the different
    importance of each parameter.

    For the current score, I just use it without any modification, as well as
    the distance to closet food. Because it will take cost for pacman to get to
    the closet food. So I think the coefficient of distance to closest food could
    just be 1.

    As for the distance to the closest normal ghost, I take the inverse of it and
    multiply it by -3 which means the score is negatively correlated with the
    distance to normal ghost. In other words, the closer the pacman is to the
    normal ghost, the situation is worse for the pacman and vice versa. When I
    apply this method, I got zero division error, then I noticed that sometimes
    the distance to closest normal ghost could be 0 which means the pacman hit
    the normal ghost and would lost almost 500 points for that behavior. So when
    that happens, I would change to distance to 0.05 instead of 0. Then the score
    of the evaluation function would undergo a drastic decrease to a very low score
    just like the real situation in the game. At first, I change 0 to 0.006 instead
    of 0.05 because 3 * 1/0.006 is closer to 500, but I found it is not stable. So
    I changed it from 0.006 to 0.01, then all the way upto 0.05 to tolerate more
    unpredictable unstabilities.

    As for distance to the closest scared ghost, it's same as the normal ghost
    except that I do not take the inverse which means this feature is positively
    correlated with the evaluated score.

    The next is number of capsule left, I use 20 because every time the pacman eat
    a scared ghost it will get 200 point while eat a food only get 10 points. This
    sound reasonable but the truth is that I adjust it mannually from 10 to 20 for
    the best performance.

    The last parameter is number of food left. The more food left, the less score
    the evaluation function would return. So the pacman need to eat as much food as
    it could. This issue was mentioned in question 2 where my pacman often swing around
    in some places without making any progress. So I count the number of food left into
    argument of evaluation function. 5 is also a result of manual modification where I
    change it from 2 to 5 for better performance.
    """
    score = currentGameState.getScore() + \
        - closest_food_dist + \
        -3 * (1.0 / closest_normal_ghost_dist) + \
        -3 * closest_scared_ghost_dist + \
        -20 * num_capsule + \
        -5 * num_food
    return score


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
