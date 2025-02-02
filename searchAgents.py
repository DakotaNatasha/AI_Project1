# searchAgents.py
# ---------------
# Source code for CS4013/5013--P1 
# ------------------------------------------------------------------
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


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from typing import List, Tuple, Any
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import pacman

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        if self.actions == None:
            self.actions = []
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    """

    def __init__(self, startingGameState: pacman.GameState):
        """
        Stores the walls, Pacman's starting position, and the four corners of the maze.
        """
        self.walls = startingGameState.getWalls()  # Store wall locations in a grid
        self.startingPosition = startingGameState.getPacmanPosition()  # Store Pacman's start position

        # Define the four corners of the maze
        top, right = self.walls.height - 2, self.walls.width - 2
        self.corners = ((1,1), (1,top), (right,1), (right,top))

        # Check if there is food in all corners and warn if missing
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))

        self._expanded = 0  # DO NOT CHANGE; Counts the number of nodes expanded during search

    def getStartState(self):
        """
        Returns the start state, represented as a tuple:
        (Pacman's current position, visitedCorners)

        Initially, Pacman has visited no corners, so visitedCorners is an empty tuple.
        """
        return (self.startingPosition, ())
        util.raiseNotDefined()

    def isGoalState(self, state: Any):
        """
        The goal state is reached when Pacman has visited all four corners.
        
        The state consists of:
        - Pacman's current position (ignored here)
        - A tuple `visitedCorners` that tracks which corners have been visited

        Goal is reached when `visitedCorners` contains all four corners.
        """
        _, visitedCorners = state
        return len(visitedCorners) == 4  # True if all four corners have been visited
        util.raiseNotDefined()

    def getSuccessors(self, state: Any):
        """
        Generates successor states by moving in all four possible directions.

        Each successor contains:
        - New position (x, y)
        - Updated `visitedCorners` tuple if a new corner is reached
        - A step cost of 1 (since every move costs 1)

        The function ensures Pacman doesn't walk into walls.
        """

        successors = []
        currentPosition, visitedCorners = state  # Unpack the current state

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Determine new position after taking action
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)

            # Check if the new position is a wall
            if not self.walls[nextx][nexty]:  
                newVisitedCorners = visitedCorners  # Default: Keep the same visited corners

                # If the new position is an unvisited corner, add it to visitedCorners
                if (nextx, nexty) in self.corners and (nextx, nexty) not in visitedCorners:
                    newVisitedCorners = visitedCorners + ((nextx, nexty),)

                # Create new state and add it to successors
                newState = ((nextx, nexty), newVisitedCorners)
                successors.append((newState, action, 1))  # Cost of moving is always 1

        self._expanded += 1  # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Computes the cost of a sequence of actions.

        If any action leads to an illegal move (hitting a wall), return a large cost (999999).
        Otherwise, return the number of actions taken (since each step has cost 1).
        """
        if actions is None: 
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: 
                return 999999  # If Pacman tries to move into a wall, return a high cost

        return len(actions)  # The cost is simply the number of actions taken




def cornersHeuristic(state: Any, problem: CornersProblem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible.
    """
    def cornersHeuristic(state: Any, problem: CornersProblem):
        """
        A heuristic for the CornersProblem.

        It estimates the minimum cost to visit all unvisited corners by:
        1. Finding the Manhattan distance to the nearest unvisited corner.
        2. Computing a Minimum Spanning Tree (MST) over remaining unvisited corners.

        The MST ensures a lower bound on the cost to complete the problem.
        """

    from util import manhattanDistance
    import itertools

    current_position, visited_corners = state

    # Identify unvisited corners
    unvisited = [corner for corner in problem.corners if corner not in visited_corners]

    if not unvisited:  
        return 0  # If all corners are visited, heuristic cost is 0 (goal state)

    # Compute nearest corner distance (for initial movement)
    nearest_corner_dist = min(manhattanDistance(current_position, corner) for corner in unvisited)

    # Compute farthest corner distance (to get an upper bound estimate)
    farthest_corner_dist = max(manhattanDistance(current_position, corner) for corner in unvisited)

    # Compute MST over the unvisited corners
    def computeMST(points):
        """Computes the Minimum Spanning Tree (MST) cost using Kruskal’s Algorithm."""
        if not points:
            return 0

        edges = []
        for p1, p2 in itertools.combinations(points, 2):
            cost = manhattanDistance(p1, p2)
            edges.append((cost, p1, p2))

        # Kruskal’s algorithm for MST
        edges.sort()  # Sort edges by cost
        parent = {p: p for p in points}

        def find(p):
            while parent[p] != p:
                p = parent[p]
            return p

        mst_cost = 0
        for cost, p1, p2 in edges:
            root1, root2 = find(p1), find(p2)
            if root1 != root2:
                mst_cost += cost
                parent[root2] = root1

        return mst_cost

    # Compute MST on unvisited corners
    mst_cost = computeMST(unvisited)

    # Final heuristic: MST + scaled nearest corner distance
    return mst_cost + 0.53 * nearest_corner_dist


    # Step 4: Compute MST on unvisited corners
    mst_cost = computeMST(unvisited)

    # Step 5: Return the heuristic estimate
    return mst_cost + 0.53 * nearest_corner_dist



class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem where the goal is to collect all food (dots) in the maze.

    The state representation consists of:
    - `pacmanPosition`: The current position of Pacman (x, y).
    - `foodGrid`: A grid representing the remaining food pellets (True = food present, False = no food).
    """

    def __init__(self, startingGameState: pacman.GameState):
        """
        Initializes the problem by storing:
        - Pacman's starting position.
        - A grid of remaining food.
        - A grid of walls.
        """
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0  # DO NOT CHANGE; Tracks number of nodes expanded in search
        self.heuristicInfo = {}  # Dictionary for heuristic function to store reusable data

    def getStartState(self):
        """
        Returns the initial search state.
        - This includes Pacman's position and the food grid at the start.
        """
        return self.start

    def isGoalState(self, state):
        """
        The goal is reached when all food has been eaten.

        - `state[1]` contains the foodGrid.
        - `foodGrid.count()` returns the number of remaining food pellets.
        - If `count() == 0`, there is no food left, meaning we reached the goal.
        """
        return state[1].count() == 0

    def getSuccessors(self, state):
        """
        Generates successor states by moving in all four possible directions.

        Each successor contains:
        - New position (x, y).
        - Updated `foodGrid` (with food removed if Pacman eats it).
        - A step cost of 1 (since every move costs 1).

        Walls are avoided to prevent illegal moves.
        """

        successors = []
        self._expanded += 1  # DO NOT CHANGE; Tracks number of expanded nodes

        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state[0]  # Extract Pacman's current position
            dx, dy = Actions.directionToVector(direction)  # Convert direction to movement (dx, dy)
            nextx, nexty = int(x + dx), int(y + dy)  # Compute next position

            # Check if Pacman can move to (nextx, nexty) (ensure it's not a wall)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()  # Copy current food grid
                nextFood[nextx][nexty] = False  # If Pacman moves onto food, remove it

                successors.append((( (nextx, nexty), nextFood), direction, 1))  # Add successor

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a sequence of actions.

        - If an action sequence contains an illegal move (walking into a wall), return a high cost (999999).
        - Otherwise, return the number of actions taken.
        """
        if actions is None: 
            return 999999

        x, y = self.getStartState()[0]
        cost = 0

        for action in actions:
            # Determine next position
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)

            # If Pacman moves into a wall, return a high cost
            if self.walls[x][y]: 
                return 999999  

            cost += 1  # Each move costs 1

        return cost  # Total cost is the number of actions taken


class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state: Tuple[Tuple, List[List]], problem: FoodSearchProblem):
    """
    A heuristic function for the FoodSearchProblem.

    Estimates the cost to eat all food by:
    1. Finding the **maximum Manhattan distance** from Pacman to any food pellet.
    2. Computing a **Minimum Spanning Tree (MST)** over all remaining food pellets.
    
    The MST provides a lower bound for the minimum cost needed to eat all food.
    """

    from util import manhattanDistance
    from util import PriorityQueue

    position, foodGrid = state  # Extract Pacman's position and food grid
    foodList = foodGrid.asList()  # Convert food grid into a list of food pellet coordinates

    if not foodList:  
        return 0  # If no food remains, heuristic should be 0 (goal state reached)

    # Compute the maximum distance from Pacman to any food pellet (Admissible)
    maxDistance = max(manhattanDistance(position, food) for food in foodList)

    # Compute a Minimum Spanning Tree (MST) on remaining food pellets
    def computeMST(points):
        """
        Computes the MST cost using **Prim's Algorithm**.

        The MST helps estimate the shortest path needed to eat all food.
        """
        if not points:
            return 0

        pq = PriorityQueue()
        visited = set()
        start = points[0]  # Pick an arbitrary start point
        visited.add(start)
        totalCost = 0

        # Add edges from start node to the priority queue
        for p in points[1:]:
            pq.push((start, p, manhattanDistance(start, p)), manhattanDistance(start, p))

        while len(visited) < len(points):  # Expand MST until all nodes are connected
            while not pq.isEmpty():
                u, v, cost = pq.pop()
                if v not in visited:
                    visited.add(v)
                    totalCost += cost
                    for p in points:
                        if p not in visited:
                            pq.push((v, p, manhattanDistance(v, p)), manhattanDistance(v, p))
                    break

        return totalCost  # The MST cost is a lower bound on the cost of collecting all food

    mstCost = computeMST(foodList)  # Compute MST over remaining food

    # Return the heuristic estimate: max(Pacman→food, MST(food))
    return max(mstCost, maxDistance)  



class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))

    def findPathToClosestDot(self, gameState: pacman.GameState):
        """
        Returns a path (a list of actions) to the closest food pellet.

        Uses **Breadth-First Search (BFS)** to ensure the shortest path is found.
        """

        # Step 1: Extract relevant information from the game state
        startPosition = gameState.getPacmanPosition()  # Get Pacman's current position
        food = gameState.getFood()  # Get grid of food locations
        walls = gameState.getWalls()  # Get grid of walls

        # Step 2: Define the search problem
        problem = AnyFoodSearchProblem(gameState)  # Create a search problem instance

        # Step 3: Use BFS to find the shortest path to any food pellet
        return search.bfs(problem)  
        util.raiseNotDefined()

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to the **nearest** food.

    This search problem inherits from `PositionSearchProblem`.
    """

    def __init__(self, gameState):
        """
        Stores information about:
        - The food grid.
        - Walls in the maze.
        - Pacman's starting position.
        """
        self.food = gameState.getFood()
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1  # Every move costs 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def isGoalState(self, state: Tuple[int, int]):
        """
        The goal state is reached when Pacman lands on a food pellet.
        """
        x, y = state
        return self.food[x][y]  # Returns True if food is present at (x, y)

        util.raiseNotDefined()

def mazeDistance(point1: Tuple[int, int], point2: Tuple[int, int], gameState: pacman.GameState) -> int:
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
