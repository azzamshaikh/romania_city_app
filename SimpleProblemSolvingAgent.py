"""
Azzam Shaikh
CS 534 - Artificial Intelligence
Individual Project # 2
March 3, 2024

This script contains the code required to run the problem-solving agent. A lot of the sample code implemented
below has been extracted from the textbook source code (https://github.com/aimacode/aima-python). All code used from
the source code will be acknowledged as per the Honor Code.
"""

import sys
from utils import *

"""
The following Problem class has been implemented from the AIMA repo. The purpose of this class is to provide a general
framework to define the problem to solve. This class will be subclassed and the actual functions like actions, result,
etc. will be implemented. 
"""


class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self. goal or checks for state in self. goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self. goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


"""
The following GraphProblem class has been implemented from the AIMA repo. This class subclasses the Problem class 
described above and defines various functions such as actions, results, and values. This GraphProblem will be utilized
as an input problem for the search algorithms to use.
"""


class GraphProblem(Problem):
    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph
        self.current = None

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                return int(distance(locs[node], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return np.inf

    def value(self, state):
        return -1 * self.h(state)


"""
The following Node class has been implemented from the AIMA repo. This class contains data about every node during the 
search process.
"""


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


"""
The following Graph class has been implemented from the AIMA repo. This class is used to create a graph out of a set of
points. In this case, the romania map data will get converted to a Graph object that can then be used in GraphProblem
object. 
"""


class Graph:
    """A graph connects nodes (vertices) by edges (links). Each edge can also
    have a length associated with it. The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2}})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C. You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added. You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B. 'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)


"""
The following UndirectedGraph function has been implemented from the AIMA repo. This function is used to convert an 
undirected graph dictionary.
"""


def UndirectedGraph(graph_dict=None):
    """Build a Graph where every edge (including future ones) goes both ways."""
    return Graph(graph_dict=graph_dict, directed=False)


"""
The following best_first_graph_search function has been implemented from the AIMA repo. This is the base algorithm used
to conduct the searches. This algorithm can have different functions input based on the value of f. Thus, for the 
implementation of greedy breadth first search and A* search, the f value will be adjusted via a lambda function. The 
function below will be called in the ProblemSolvingAgent.

For greedy best first search, the f function is a heuristic estimate to the goal. 

For A* search, the f function is a heuristic estimate to the goal. The heuristic for this search algorithm is defined
as the sum of the cost of the path from the start position to the current node and the heuristic function that 
finds the cheapest path from the current node to the goal. 
"""


def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node. depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    the best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None


"""
The following hill climbing function has been implemented from the AIMA repo. This is the algorithm used to conduct the 
hill climbing search. The function below will be called in the ProblemSolvingAgent.
"""


def hill_climbing(problem):
    """
    [Figure 4.2]
    From the initial node, keep choosing the neighbor with the highest value,
    stopping when no neighbor is better.
    """
    current = Node(problem.initial)
    while True:
        neighbors = current.expand(problem)
        if not neighbors:
            break
        neighbor = argmax_random_tie(neighbors, key=lambda node: problem.value(node.state))
        if problem.value(neighbor.state) <= problem.value(current.state):
            break
        current = neighbor
    if current.state == problem.initial:
        return None
    else:
        return current


"""
The following exp_schedule and simulated_annealing functions have been implemented from the AIMA repo. This is the 
algorithm used to conduct the simulated annealing search. The functions below will be called in the ProblemSolvingAgent.
"""


def exp_schedule(k=20, lam=0.005, limit=100):
    """One possible schedule function for simulated annealing"""
    return lambda t: (k * np.exp(-lam * t) if t < limit else 0)


def simulated_annealing(problem, schedule=exp_schedule()):
    """[Figure 4.5] CAUTION: This differs from the pseudocode as it
    returns a state instead of a Node."""
    current = Node(problem.initial)
    for t in range(sys.maxsize):
        T = schedule(t)
        if T == 0:
            return current
        neighbors = current.expand(problem)
        if not neighbors:
            return current
        next_choice = random.choice(neighbors)
        delta_e = problem.value(next_choice.state) - problem.value(current.state)
        if delta_e > 0 or probability(np.exp(delta_e / T)):
            current = next_choice


"""
The following SimpleProblemSolvingAgentProgram class has been implemented from the AIMA repo. This class will create an
SPSA object that will solve the GraphProblem. The abstract class has been modified to output the appropriate information
for IP #2.
"""


class SimpleProblemSolvingAgentProgram:
    """
    [Figure 3.1]
    Abstract framework for a problem-solving agent.
    """
    def __init__(self, initial_state, input_map, location):
        """State is an abstract representation of the state
        of the world, and seq is the list of actions required
        to get to a particular state from the initial state(root).

        Upon initialization of the SPSA object, the initial state,
        input map, and input map locations will be stored.
        """
        self.initial_state = initial_state
        self.state = None
        self.seq = []
        self.map = input_map
        self.map.locations = location
        self.methods = ["greedy", "astar", "hill", "annealing"]

    def __call__(self, percept):
        """[Figure 3.1] Formulate a goal and problem, then
        search for a sequence of actions to solve it.

        Upon calling the SPSA object, a percept, or goal, will be
        given. This goal will get fed to the run_problem_solving_agent
        function while will run the SPSA program with the different methods
        required for IP #2.
        """
        for method in self.methods:
            self.run_problem_solving_agent(percept, method)

    def run_problem_solving_agent(self, percept, method):
        """
        This function gets called whenever the SPSA object gets called.
        The goal will be formulated, the problem will be defined, and
        then search will occur. The search output will then get printed
        according to the requirements in IP #2.
        """
        self.state = self.update_state(percept)
        if not self.seq:
            goal = self.formulate_goal(self.state)
            problem = self.formulate_problem(self.initial_state, goal)
            self.seq = self.search(problem, method)
        self.print_output(method, self.seq)
        self.seq = []

    def update_state(self, percept):
        """
        Updates the current state based on the percept.
        """
        return percept

    def formulate_goal(self, state):
        """
        Formulates the goal for the problem based on the state.
        In this case, the goal would be the state that is input.
        """
        return state

    def formulate_problem(self, initial_state, goal):
        """
        Formulates the problem for the SPSA. The problem would be
        defined as a GraphProblem object with a given initial state,
        goal state, and input map.
        """
        return GraphProblem(initial_state, goal, self.map)

    def search(self, problem, method):
        """
        Searches for the solution given the GraphProblem. Since the
        IP #2 requires 4 different search methods, when the SPSA object
        gets called, 4 different methods will be applied to search for the
        solution. Thus, the switch statement allows for custom search functions
        based on the different methods.
        """
        match method:
            case "greedy":
                h = memoize(problem.h, 'h')
                return best_first_graph_search(problem, lambda n: h(n), display=False)
            case "astar":
                h = memoize(problem.h, 'h')
                return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display=False)
            case "hill":
                return hill_climbing(problem)
            case "annealing":
                return simulated_annealing(problem, exp_schedule())

    def print_output(self, method, node: Node):
        """
        Once the search function solves the problem, the node output from the
        search function will get sent to this function and the various data
        required to be output as per IP #2 requirements will get printed.
        """
        if node is None:
            print()
            self.print_name(method)
            print('There is no solution found with this algorithm.')
        else:
            print()
            self.print_name(method)
            output = ' \u2192 '.join(node.solution())
            print(self.initial_state + ' \u2192 ' + output)
            print('Total Cost: ' + str(node.path_cost))

    @staticmethod
    def print_name(method):
        """
        A supporting function to print out the function name based on the
        method used.
        """
        match method:
            case "greedy":
                print('Greedy Best-First Search')
            case "astar":
                print('A* Search')
            case "hill":
                print('Hill Climbing Search')
            case "annealing":
                print('Simulated Annealing Search')


def main():
    """
    This main function is used for troubleshooting and code verification.
    This allows this SimpleProblemSolvingAgent.py file to be run as an
    executable. The parameters and inputs below are hard coded for testing.

    Run the RomaniaCityApp.py file as an executable to run the main program.
    """

    romania_map = UndirectedGraph(dict(
        Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
        Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
        Craiova=dict(Drobeta=120, Rimnicu=146, Pitesti=138),
        Drobeta=dict(Mehadia=75),
        Eforie=dict(Hirsova=86),
        Fagaras=dict(Sibiu=99),
        Hirsova=dict(Urziceni=98),
        Iasi=dict(Vaslui=92, Neamt=87),
        Lugoj=dict(Timisoara=111, Mehadia=70),
        Oradea=dict(Zerind=71, Sibiu=151),
        Pitesti=dict(Rimnicu=97),
        Rimnicu=dict(Sibiu=80),
        Urziceni=dict(Vaslui=142)))
    romania_map.locations = dict(
        Arad=(91, 492), Bucharest=(400, 327), Craiova=(253, 288),
        Drobeta=(165, 299), Eforie=(562, 293), Fagaras=(305, 449),
        Giurgiu=(375, 270), Hirsova=(534, 350), Iasi=(473, 506),
        Lugoj=(165, 379), Mehadia=(168, 339), Neamt=(406, 537),
        Oradea=(131, 571), Pitesti=(320, 368), Rimnicu=(233, 410),
        Sibiu=(207, 457), Timisoara=(94, 410), Urziceni=(456, 350),
        Vaslui=(509, 444), Zerind=(108, 531))

    spsa = SimpleProblemSolvingAgentProgram('Arad', romania_map, romania_map.locations)
    spsa('Bucharest')


if __name__ == "__main__":
    """
    Allows the SimpleProblemSolvingAgent.py file to be ran as 
    an individual script. 
    """
    main()


