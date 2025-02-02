# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    from util import Stack  # Import the stack from util.py

    # Initialize the frontier with the state of the start node
    frontier = Stack() # Creates an empty stack that will store nodes still needing explored- THIS IS THE FRINGE
    frontier.push((problem.getStartState(), [], 0))  # (state, path, cost)
    """
        problem.getStartState() gets the starting position of Pacman in the maze
        frontier.push((state, path, cost)) adds the start node to the stack (fringe)
        state is the current pacman position or the current node, this is represented using (x,y) coordinates
        path is a list containing the pacman's directional (north, south, east, west, stop) moves 
        cost collects the cost of each move (should be 1 for each box the pacman traverses)
    """
    
    # This tracks the explored nodes so we don't expand the same node twice and cause an infinte loop if a cyclic graph is used
    visited = set() 

    """
    This while loop performs this function until the fringe stack containing the nodes is empty
    While the stack is not empty and has something in it...
    """
    while not frontier.isEmpty(): 
        state, path, cost = frontier.pop() 
        """ 
        First we will pop the first node from the stack. 
        When we pop this node from the stack we are storing its infromation
            -the state variable contains its node index which describes the current possition of the pacman
            -the path variable contains a directional action taken by the pacman to reach this node (N, S, E, W, stop)
            -its cost variable contains the cost for traversing to this node (1 unless the terrain differs)
        These are stored because the agent needs the index, the direction, and the cost to identify the path for the goal
        (since this is DFS it adheres to last in first out order)
        """
       
        
        # If this node is the goal, return the path to get here
        if problem.isGoalState(state):
            return path
        
        #If this node has not been explored before add it to the visited set to prevent revisiting nodes
        if state not in visited:
            visited.add(state)  
            
            # Steps taken for exploring the current node
            for successor, action, stepCost in problem.getSuccessors(state):
                """
                successor is the next possible node or also called the child node 
                action is the next move needed to reach the next possible node (successor)
                stepCost is the cost of moving to the next possible state (successor)
                """
                
                #if this node (successor) hasn't been explored push this unvisited node onto the stack
                if successor not in visited:
                    
                    #creates a new path that appends this action onto the old path making it the current path
                    newPath = path + [action] 
                   
                    #push the newpath and the increased cost of this successor and the index of the successor onto the stack
                    frontier.push((successor, newPath, cost + stepCost))
                    
    return []  # If no solution is found (shouldn't happen in a solvable maze)
    #if the fringe stack is empty and the goal is not found no solution exists and the function will return an empty list without a path.
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the shallowest nodes in the search tree first.
    
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    from util import Queue  # Import the queue from util.py

    # Initialize the frontier with the start state
    frontier = Queue()  # Creates an empty queue that will store nodes still needing exploration - THIS IS THE FRINGE
    frontier.push((problem.getStartState(), [], 0))  # (state, path, cost)
    """
        problem.getStartState() gets the starting position of Pacman in the maze
        frontier.push((state, path, cost)) adds the start node to the queue (fringe)
        state is the current pacman position or the current node, represented using (x,y) coordinates
        path is a list containing the pacman's directional (north, south, east, west, stop) moves 
        cost collects the cost of each move (should be 1 for each box the pacman traverses)
    """
    
    # This tracks the explored nodes so we don't expand the same node twice, preventing infinite loops in cyclic graphs
    visited = set()  

    """
    This while loop executes until the queue containing the nodes is empty
    While the queue is not empty and has something in it...
    """
    while not frontier.isEmpty(): 
        state, path, cost = frontier.pop()  
        """ 
        First, we dequeue (pop) the first node from the queue. 
        When we remove this node from the queue, we store its information:
            - The `state` variable contains its node index, which describes the current position of Pacman
            - The `path` variable contains a list of directional actions taken by Pacman to reach this node (N, S, E, W, stop)
            - The `cost` variable contains the cost for traversing to this node (1 unless the terrain differs)
        These are stored because the agent needs the node index, the direction, and the cost to track the optimal path.
        (Since this is BFS, it adheres to **first in, first out (FIFO)** order)
        """
       
        # If this node is the goal, return the path to get here
        if problem.isGoalState(state):
            return path

        # If this node has not been visited before, add it to the visited set to prevent revisiting nodes
        if state not in visited:
            visited.add(state)  
            
            # Expand the current node
            for successor, action, stepCost in problem.getSuccessors(state):
                """
                successor is the next possible node or also called the child node 
                action is the next move needed to reach the next possible node (successor)
                stepCost is the cost of moving to the next possible state (successor)
                """
                
                # If this node (successor) hasn't been visited, enqueue(add/push) it onto the queue
                if successor not in visited:
                    
                    # Create a new path that appends this action onto the old path, making it the current path
                    newPath = path + [action] 
                    
                    # Enqueue the new path, the increased cost of this successor, and the index of the successor onto the queue
                    frontier.push((successor, newPath, cost + stepCost))
                    
    return []  # If no solution is found (shouldn't happen in a solvable maze)
    # If the fringe queue is empty and the goal is not found, no solution exists and the function will return an empty list without a path.
    util.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the node of least total cost first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    from util import PriorityQueue  # Import the priority queue from util.py

    # Initialize the frontier with the start state
    frontier = PriorityQueue()  # Creates an empty priority queue that will store nodes to be explored - THIS IS THE FRINGE
    frontier.push((problem.getStartState(), [], 0), 0)  # (state, path, cost), priority)
    """
        problem.getStartState() gets the starting position of Pacman in the maze
        frontier.push((state, path, cost), priority) adds the start node to the priority queue (fringe)
        state is the current pacman position or the current node, represented using (x,y) coordinates
        path is a list containing the pacman's directional (north, south, east, west, stop) moves 
        cost collects the cost of each move (should be 1 for each box the pacman traverses unless the terrain differs)
        The priority is now accounted for. The prioprity of a node is determined by the total path cost up to that node.
    """

    # This tracks the explored nodes so we don't expand the same node twice, preventing redundant expansions
    visited = set()

    """
    This while loop performs UCS until the priority queue is empty
    While the priority queue is not empty and has something in it...
    """
    while not frontier.isEmpty():
        state, path, cost = frontier.pop()  # Pop the least-cost node
        """ 
        First, we dequeue (pop) the node with the LOWEST TOTAL PATH COST from the priority queue.
        When we remove this node from the queue, we store its information:
            - `state`: The node index, which describes the current position of Pacman
            - `path`: A list of directional actions taken by Pacman to reach this node (N, S, E, W, stop)
            - `cost`: The cumulative cost to reach this node
        (Since this is UCS, it expands nodes in order of lowest accumulated cost.)
        """

        # If this node is the goal, return the path to get here
        if problem.isGoalState(state):
            return path

        # If this node has not been visited before, add it to the visited set to prevent revisiting nodes
        if state not in visited:
            visited.add(state)

            # Expand the current node by exploring its successors
            for successor, action, stepCost in problem.getSuccessors(state):
                """
                successor is the next possible node or also called the child node 
                action is the next move needed to reach the next possible node (successor)
                stepCost is the cost of moving to the next possible state (successor)
                """
                
                # If this node (successor) hasn't been visited, enqueue it onto the priority queue with updated cost
                if successor not in visited:
                    newPath = path + [action]  # Create a new path that extends the previous path
                    newCost = cost + stepCost  # Update the total cost to reach this successor
                    
                    # Push the successor into the priority queue with priority = total cost so far
                    frontier.push((successor, newPath, newCost), newCost)

    return []  # If no solution is found (shouldn't happen in a solvable maze)
    # If the priority queue is empty and the goal is not found, no solution exists and the function will return an empty list without a path.
    util.raiseNotDefined()


def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """
    Search the node that has the lowest combined cost and heuristic first.

    A* Search uses:
    - `g(n)`: The actual cost from the start state to the current state.
    - `h(n)`: The heuristic estimate from the current state to the goal.
    - `f(n) = g(n) + h(n)`: The total estimated cost of the cheapest solution passing through `n`.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    from util import PriorityQueue  # Import the priority queue from util.py

    # Initialize the frontier with the start state
    frontier = PriorityQueue()  # Priority queue stores nodes sorted by f(n) = g(n) + h(n)
    start_state = problem.getStartState()
    
    # Push initial state with priority = g(n) + h(n)
    frontier.push((start_state, [], 0), heuristic(start_state, problem))
    """
        problem.getStartState() gets the starting position of Pacman in the maze
        frontier.push((state, path, cost), priority) adds the start node to the priority queue (fringe)
        state is the current pacman position or the current node, represented using (x,y) coordinates
        path is a list containing the pacman's directional (north, south, east, west, stop) moves 
        cost collects the cost of each move (should be 1 for each box the pacman traverses)
        The priority of a node is determined by f(n) = g(n) + h(n)
    """

    # This dictionary tracks visited nodes and their best known cost
    visited = {}

    """
    This while loop performs A* until the priority queue is empty
    While the priority queue is not empty and has something in it...
    """
    while not frontier.isEmpty():
        state, path, cost = frontier.pop()  # Get the node with the lowest f(n)
        """ 
        First, we dequeue (pop) the node with the **lowest f(n) = g(n) + h(n)** from the priority queue.
        When we remove this node from the queue, we store its information:
            - `state`: The node index, which describes the current position of Pacman
            - `path`: A list of directional actions taken by Pacman to reach this node (N, S, E, W, stop)
            - `cost`: The cumulative path cost to reach this node (g(n))
        (Since this is A*, it **expands nodes in order of lowest estimated total cost**.)
        """

        # If this node is the goal, return the path to get here
        if problem.isGoalState(state):
            return path

        # If we have already reached this state with a lower cost, we skip processing it
        if state in visited and cost >= visited[state]:  
            continue  
        visited[state] = cost  # Mark this cost as the best known path

        # Expand the current node by exploring its successors
        for successor, action, stepCost in problem.getSuccessors(state):
            """
            successor is the next possible node or also called the child node 
            action is the next move needed to reach the next possible node (successor)
            stepCost is the cost of moving to the next possible state (successor)
            """
            
            # Compute the new path and cost
            newPath = path + [action]  # Create a new path that extends the previous path
            newCost = cost + stepCost  # Update the total cost to reach this successor (g(n))

            # Compute f(n) = g(n) + h(n), where h(n) is the heuristic estimate of remaining cost
            priority = newCost + heuristic(successor, problem)

            # Push the successor into the priority queue with priority = f(n)
            frontier.update((successor, newPath, newCost), priority)

    return []  # If no solution is found (shouldn't happen in a solvable maze)
    # If the priority queue is empty and the goal is not found, no solution exists and the function will return an empty list without a path.
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
