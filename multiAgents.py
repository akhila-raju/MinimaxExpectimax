# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghostVal= 0
        nearestFood = 0
        ghostArray= []
        foodArray = []
        foodList = newFood.asList()
        ghostPos = successorGameState.getGhostPositions()
        leftOverFood = successorGameState.getNumFood()

        for elem in foodList:
            foodArray.append(manhattanDistance(elem, newPos))
        for ghost in newGhostStates:
            if ghostState.scaredTimer == 0:
                ghostArray.append(manhattanDistance(newPos, ghost.getPosition()))
        if foodArray != []:
            nearestFood = min(foodArray)
        for ghost in newGhostStates:
            ghostClosest = manhattanDistance(newPos, ghost.getPosition())
            if ghost.scaredTimer > ghostClosest:
                ghostVal += ghost.scaredTimer - ghostClosest
        if ghostArray != []:
            ghostVal += min(ghostArray)
        weighted_sum = ghostVal - nearestFood - 15*leftOverFood

        return weighted_sum

        
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        return self.val(gameState, numAgents, self.depth+1)[1]


    def val(self, gameState, numAgents, currDepth, agentIndex=0):
      """
      Runs minimax algorithm and returns value and an action.
      """
      if currDepth == 0 or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState), None

      if agentIndex == 0:
        currDepth -= 1
        max_val = self.max_val(gameState, numAgents, currDepth, agentIndex)
        return max_val
      else:
        min_val = self.min_val(gameState, numAgents, currDepth, agentIndex)
        return min_val

    def max_val(self, gameState, numAgents, currDepth, agentIndex):
      """
      Returns max value and an action.
      """
      val, action = -100000, None
      currVal = -100000
      # This will generate several actions, each of which has its own successor.
      legalActions = gameState.getLegalActions(agentIndex)
      if len(legalActions) == 0 or currDepth == 0:
        return self.evaluationFunction(gameState), None
      # Generate action and value for each next game state (successor).
      for currAction in legalActions:
        successor = gameState.generateSuccessor(agentIndex, currAction)
        currVal = max(val, self.val(successor, numAgents, currDepth, (agentIndex+1) % numAgents)[0])

        if currVal > val:
          val, action = currVal, currAction

      return (val, action)


    def min_val(self, gameState, numAgents, currDepth, agentIndex):
      """
      Returns min value and an action.
      """
      val, action = 100000, None
      currVal = 100000
      # This will generate several actions, each of which has its own successor.
      legalActions = gameState.getLegalActions(agentIndex)
      if len(legalActions) == 0 or currDepth == 0:
        return self.evaluationFunction(gameState), None
      # Generate action and value for each next game state (successor).
      for currAction in legalActions:
        successor = gameState.generateSuccessor(agentIndex, currAction)
        currVal = min(val, self.val(successor, numAgents, currDepth, (agentIndex+1) % numAgents)[0])

        if currVal < val:
          val, action = currVal, currAction

      return (val, action)



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        return self.val(gameState, numAgents, self.depth+1)[1]


    def val(self, gameState, numAgents, currDepth, agentIndex=0, alpha=-100000, beta=100000):
      """
      Runs minimax algorithm and returns value and an action.
      """
      if currDepth == 0 or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState), None

      if agentIndex == 0:
        currDepth -= 1
        max_val = self.max_val(gameState, numAgents, currDepth, agentIndex, alpha, beta)
        return max_val
      else:
        min_val = self.min_val(gameState, numAgents, currDepth, agentIndex, alpha, beta)
        return min_val


    def max_val(self, gameState, numAgents, currDepth, agentIndex, alpha, beta):
      """
      Returns max value and an action.
      """
      val, action = -100000, None
      currVal = -100000
      # This will generate several actions, each of which has its own successor.
      legalActions = gameState.getLegalActions(agentIndex)
      if len(legalActions) == 0 or currDepth == 0:
        return self.evaluationFunction(gameState), None
      # Generate action and value for each next game state (successor).
      for currAction in legalActions:
        successor = gameState.generateSuccessor(agentIndex, currAction)
        currVal = max(val, self.val(successor, numAgents, currDepth, (agentIndex+1) % numAgents, alpha, beta)[0])

        if currVal > val:
          val, action = currVal, currAction

        if currVal > beta:
          return currVal, currAction

        alpha = max(alpha, currVal)

      return (val, action)


    def min_val(self, gameState, numAgents, currDepth, agentIndex, alpha, beta):
      """
      Returns min value and an action.
      """
      val, action = 100000, None
      currVal = 100000
      # This will generate several actions, each of which has its own successor.
      legalActions = gameState.getLegalActions(agentIndex)
      if len(legalActions) == 0 or currDepth == 0:
        return self.evaluationFunction(gameState), None
      # Generate action and value for each next game state (successor).
      for currAction in legalActions:
        successor = gameState.generateSuccessor(agentIndex, currAction)
        currVal = min(val, self.val(successor, numAgents, currDepth, (agentIndex+1) % numAgents, alpha, beta)[0])

        if currVal < val:
          val, action = currVal, currAction

        if currVal < alpha:
          return currVal, currAction

        beta = min(beta, currVal)

      return (val, action)


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
        numAgents = gameState.getNumAgents()
        return self.val(gameState, numAgents, self.depth+1)[1]


    def val(self, gameState, numAgents, currDepth, agentIndex=0):
      """
      Runs minimax algorithm and returns value and an action.
      """
      if currDepth == 0 or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState), None

      if agentIndex == 0:
        currDepth -= 1
        max_val = self.max_val(gameState, numAgents, currDepth, agentIndex)
        return max_val
      else:
        exp_val = self.exp_val(gameState, numAgents, currDepth, agentIndex)
        return exp_val

    def max_val(self, gameState, numAgents, currDepth, agentIndex):
      """
      Returns max value and an action.
      """
      val, action = -100000, None
      currVal = -100000
      # This will generate several actions, each of which has its own successor.
      legalActions = gameState.getLegalActions(agentIndex)
      if len(legalActions) == 0 or currDepth == 0:
        return self.evaluationFunction(gameState), None
      # Generate action and value for each next game state (successor).
      for currAction in legalActions:
        successor = gameState.generateSuccessor(agentIndex, currAction)
        currVal = max(val, self.val(successor, numAgents, currDepth, (agentIndex+1) % numAgents)[0])

        if currVal > val:
          val, action = currVal, currAction

      return (val, action)


    def exp_val(self, gameState, numAgents, currDepth, agentIndex):
      """
      Returns expected value and an action.
      """
      val, action = 0, None
      currVal = 0
      # This will generate several actions, each of which has its own successor.
      legalActions = gameState.getLegalActions(agentIndex)
      if len(legalActions) == 0 or currDepth == 0:
        return self.evaluationFunction(gameState), None
      # Generate action and value for each next game state (successor).
      for currAction in legalActions:
        successor = gameState.generateSuccessor(agentIndex, currAction)
        currVal += float(self.val(successor, numAgents, currDepth, (agentIndex+1) % numAgents)[0])

      val = currVal / float(len(legalActions))

      return (val, action)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <we essentially tweaked the logic in our eval function from above.
                    The main change was in our weighted sum, where we ignored the number of food left in our sum
                    and factored in the current game score. Lastly, instead of food left, we wanted to maximize the weight for capsules.
                    >
    """
    "*** YOUR CODE HERE ***"
    currPos = currentGameState.getPacmanPosition()
    currFoodList = currentGameState.getFood().asList()
    currGhostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    numPowerCapsules = len(capsules)
    ghostVal= 0
    nearestFood = 0
    ghostArray= []
    foodArray = []

    for elem in currFoodList:
        foodArray.append(-manhattanDistance(elem, currPos))
    for ghost in currGhostStates:
        if ghost.scaredTimer == 0:
            ghostArray+=[1]
        ghostArray.append(-manhattanDistance(currPos, ghost.getPosition()))
    if foodArray == []:
        foodArray+=[1]
        nearestFood = max(foodArray)
    if ghostArray != []:
        ghostVal += min(ghostArray)
    weighted_sum = currentGameState.getScore() + nearestFood + ghostVal - 20*numPowerCapsules

    return weighted_sum

# Abbreviation
better = betterEvaluationFunction