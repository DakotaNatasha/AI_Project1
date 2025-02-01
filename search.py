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

    # Initialize the frontier with the start state
    frontier = Stack() # Creates an empty stack that will store nodes still needing explored
    frontier.push((problem.getStartState(), [], 0))  # (state, path, cost)
    #problem.getStartState() gets the starting position of Pacman in the maze
    #frontier.push((state, path, cost)) adds the start node to the stack
    #state is the current pacman position, this is represented using (x,y) coordinates
    #path is a list detailing the pacman's moves
    #cost collects the cost of each move
    
    visited = set()  # Track visited states so we don't expand the same state twice and cause an infinte loop if a cyclic graph is used

    #This while loop loops the stack containing the nodes in the frontier untill this stack is empty 
    while not frontier.isEmpty():
        state, path, cost = frontier.pop()
        #this pops the top node from the stack (last in first out)
        #the current possition of the pacman, the list of actions taken to reach this state, and the total cost to reach the state is all of the information popped from the stack
        
        # If this state is the goal, return the path to get here
        if problem.isGoalState(state):
            return path
        
        #If this state has not been visited before add it to the visited set to prevent revisiting nodes and reduce redundancies
        if state not in visited:
            visited.add(state)  # Mark node as visited
            
            # Expand the current node
            #successor is the next possible state
            #action is the next move needed to reach the next possible state (successor)
            #stepCost is the cost of moving to the next possible state (successor)
            for successor, action, stepCost in problem.getSuccessors(state):
                
                #if this state (successor) hasn't been visited push unvisited it onto the stack
                if successor not in visited:
                    newPath = path + [action] 
                    #creates a new path that appends this action onto the old path making it the current path
                    frontier.push((successor, newPath, cost + stepCost))
                    #pushes the newpath and the increased cost of this successor onto the stack

    return []  # If no solution is found (shouldn't happen in a solvable maze)
    #if the stack is empty and the goal is not found no solution exists and the function will return an empty list without a path.
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
