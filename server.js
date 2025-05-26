const express = require('express');
const app = express();
const PORT = 3000;
const cors = require('cors');
app.use(cors());
// Example GET route: /get-letter?filename=example.jpg
const info1 = `
class State:
    def __init__(self, jug1, jug2):
        self.jug1 = jug1
        self.jug2 = jug2

    def __eq__(self, other):
        return self.jug1 == other.jug1 and self.jug2 == other.jug2

    def __hash__(self):
        return hash((self.jug1, self.jug2))

    def __str__(self):
        return f"({self.jug1}, {self.jug2})"


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

    def path(self):
        if self.parent is None:
            return [self.state]
        else:
            return self.parent.path() + [self.state]


def dfs(start_state, goal):
    visited = set()
    stack = [Node(start_state)]

    while stack:
        node = stack.pop()
        state = node.state

        if state == goal:
            return node.path()

        visited.add(state)

        actions = [
            (state.jug1, 4),  # Fill Jug2
            (4, state.jug2),  # Fill Jug1
            (0, state.jug2),  # Empty Jug1
            (state.jug1, 0),  # Empty Jug2
            (min(state.jug1 + state.jug2, 4), max(0, state.jug1 + state.jug2 - 4)),  # Pour Jug1 -> Jug2
            (max(0, state.jug1 + state.jug2 - 3), min(state.jug1 + state.jug2, 3))   # Pour Jug2 -> Jug1
        ]

        for action in actions:
            new_state = State(action[0], action[1])
            if new_state not in visited:
                stack.append(Node(new_state, node))

    return None


# Test the algorithm with an example
start_state = State(0, 0)     # Initial state: both jugs are empty
goal_state = State(2, 0)      # Goal state: jug1 has 2 units of water

print("Starting DFS for Water Jug Problem...")
path = dfs(start_state, goal_state)

if path:
    print("Solution found! Steps to reach the goal:")
    for i, state in enumerate(path):
        print(f"Step {i}: Jug1: {state.jug1}, Jug2: {state.jug2}")
else:
    print("No solution found!")
`;
const info2 = `
from queue import PriorityQueue

# State representation: (left_missionaries, left_cannibals, boat_position)
INITIAL_STATE = (3, 3, 1)
GOAL_STATE = (0, 0, 0)

def is_valid_state(state):
    left_m, left_c, _ = state
    right_m = 3 - left_m
    right_c = 3 - left_c

    # Missionaries can't be outnumbered by cannibals on either side
    if (left_m > 0 and left_c > left_m) or (right_m > 0 and right_c > right_m):
        return False
    return True

def generate_next_states(state):
    next_states = []
    left_m, left_c, boat_pos = state
    new_boat_pos = 1 - boat_pos

    for m in range(3):
        for c in range(3):
            if 1 <= m + c <= 2:
                if boat_pos == 1:
                    new_left_m = left_m - m
                    new_left_c = left_c - c
                else:
                    new_left_m = left_m + m
                    new_left_c = left_c + c

                new_state = (new_left_m, new_left_c, new_boat_pos)
                if 0 <= new_left_m <= 3 and 0 <= new_left_c <= 3 and is_valid_state(new_state):
                    next_states.append(new_state)
    return next_states

def bfs():
    frontier = PriorityQueue()
    frontier.put((0, INITIAL_STATE))
    came_from = {}
    cost_so_far = {INITIAL_STATE: 0}

    while not frontier.empty():
        _, current_state = frontier.get()
        if current_state == GOAL_STATE:
            break

        for next_state in generate_next_states(current_state):
            new_cost = cost_so_far[current_state] + 1
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                frontier.put((new_cost, next_state))
                came_from[next_state] = current_state

    # Reconstruct path
    current_state = GOAL_STATE
    path = [current_state]
    while current_state != INITIAL_STATE:
        current_state = came_from[current_state]
        path.append(current_state)
    path.reverse()
    return path

def print_path(path):
    for i, state in enumerate(path):
        left_m, left_c, boat_pos = state
        right_m = 3 - left_m
        right_c = 3 - left_c
        print(f"Step {i}: ({left_m}M, {left_c}C, {'left' if boat_pos == 1 else 'right'}) "
              f"-> ({right_m}M, {right_c}C, {'right' if boat_pos == 1 else 'left'})")

if __name__ == "__main__":
    path = bfs()
    print("Solution path:")
    print_path(path)
`;
const info3 = `
import heapq

def astar(start, goal, neighbors, h):
    class Node:
        def __init__(self, s, p=None, g=0): self.s, self.p, self.g = s, p, g
        def f(self): return self.g + h(self.s)

    open_set = [(h(start), id(start), Node(start))]
    closed = set()

    while open_set:
        _, _, curr = heapq.heappop(open_set)
        if curr.s == goal:
            path = []
            while curr: path.append(curr.s); curr = curr.p
            return path[::-1]
        closed.add(curr.s)
        for ns in neighbors(curr.s):
            if ns in closed: continue
            n = Node(ns, curr, curr.g + 1)
            if any(ns == x.s for _, _, x in open_set): continue
            heapq.heappush(open_set, (n.f(), id(n), n))
    return None

neighbors = lambda s: [(s[0]+dx, s[1]+dy) for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]]
heuristic = lambda s: abs(4 - s[0]) + abs(4 - s[1])

path = astar((0, 0), (4, 4), neighbors, heuristic)
print("Path:", path)
`;
const info4 = `
import heapq

def astar(start, goal, neighbors, h):
    class Node:
        def __init__(self, s, p=None, g=0): self.s, self.p, self.g = s, p, g
        def f(self): return self.g + h(self.s)

    open_set = [(h(start), id(start), Node(start))]
    closed = set()

    while open_set:
        _, _, curr = heapq.heappop(open_set)
        if curr.s == goal:
            path = []
            while curr: path.append(curr.s); curr = curr.p
            return path[::-1]
        closed.add(curr.s)
        for ns in neighbors(curr.s):
            if ns in closed: continue
            n = Node(ns, curr, curr.g + 1)
            if any(ns == x.s for _, _, x in open_set): continue
            heapq.heappush(open_set, (n.f(), id(n), n))
    return None

neighbors = lambda s: [(s[0]+dx, s[1]+dy) for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]]
heuristic = lambda s: abs(4 - s[0]) + abs(4 - s[1])

path = astar((0, 0), (4, 4), neighbors, heuristic)
print("Path:", path)
`;
const info5 = `
def is_safe(board, row, col):
    for i in range(row):
        if board[i] == col:
            return False
    for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
        if board[i] == j:
            return False
    for i, j in zip(range(row-1, -1, -1), range(col+1, 8)):
        if board[i] == j:
            return False
    return True

def solve_queens_util(board, row):
    if row >= 8:
        return True
    for col in range(8):
        if is_safe(board, row, col):
            board[row] = col
            if solve_queens_util(board, row + 1):
                return True
            board[row] = -1
    return False

def solve_queens():
    board = [-1] * 8
    if not solve_queens_util(board, 0):
        print("Solution does not exist")
        return False
    print("Solution:")
    for i in range(8):
        for j in range(8):
            if board[i] == j:
                print("Q", end=" ")
            else:
                print(".", end=" ")
        print()
    return True

solve_queens()
`;
const info6 = `
import numpy as np

def tsp_nearest_neighbor(distances):
    num_cities = distances.shape[0]
    visited = [False] * num_cities
    tour = []
    current_city = 0
    tour.append(current_city)
    visited[current_city] = True

    for _ in range(num_cities - 1):
        nearest_city = None
        nearest_distance = float('inf')
        for next_city in range(num_cities):
            if not visited[next_city] and distances[current_city, next_city] < nearest_distance:
                nearest_city = next_city
                nearest_distance = distances[current_city, next_city]
        current_city = nearest_city
        tour.append(current_city)
        visited[current_city] = True

    tour.append(tour[0])  # Return to starting city
    return tour

if __name__ == "__main__":
    distances = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])
    tour = tsp_nearest_neighbor(distances)
    print("Tour:", tour)
`;
const info7 = `
class KnowledgeBase:
    def __init__(self):
        self.known_facts = set()
        self.inference_rules = []

    def add_fact(self, fact):
        self.known_facts.add(fact)

    def add_rule(self, condition, result):
        self.inference_rules.append((condition, result))

    def forward_chaining(self, target):
        derived_facts = set()
        to_process = list(self.known_facts)

        while to_process:
            current = to_process.pop(0)
            if current == target:
                return True

            for condition, result in self.inference_rules:
                if condition in derived_facts:
                    if result not in derived_facts and result not in to_process:
                        to_process.append(result)

            derived_facts.add(current)

        return False

if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.add_fact("A")
    kb.add_fact("B")
    kb.add_rule("A", "C")
    kb.add_rule("B", "C")
    kb.add_rule("C", "D")
    
    target_goal = "D"
    if kb.forward_chaining(target_goal):
        print(f"The goal '{target_goal}' is reachable.")
    else:
        print(f"The goal '{target_goal}' is not reachable.")
`;
const info8 = `
class Statement:
    def __init__(self, predicate_name, parameters):
        self.predicate_name = predicate_name
        self.parameters = parameters

    def __eq__(self, other):
        return isinstance(other, Statement) and self.predicate_name == other.predicate_name and self.parameters == other.parameters

    def __hash__(self):
        return hash((self.predicate_name, tuple(self.parameters)))

    def __str__(self):
        return f"{self.predicate_name}({', '.join(self.parameters)})"

    def __lt__(self, other):
        if not isinstance(other, Statement):
            return NotImplemented
        if self.predicate_name < other.predicate_name:
            return True
        elif self.predicate_name == other.predicate_name:
            return self.parameters < other.parameters
        else:
            return False

class Rule:
    def __init__(self, statements):
        self.statements = set(statements)

    def __eq__(self, other):
        return isinstance(other, Rule) and self.statements == other.statements

    def __hash__(self):
        return hash(tuple(sorted(self.statements)))

    def __str__(self):
        return " | ".join(str(stmt) for stmt in self.statements)

def apply_resolution(rule1, rule2):
    new_rules = set()
    for stmt1 in rule1.statements:
        for stmt2 in rule2.statements:
            if stmt1.predicate_name == stmt2.predicate_name and stmt1.parameters != stmt2.parameters:
                merged_statements = (rule1.statements | rule2.statements) - {stmt1, stmt2}
                new_rules.add(Rule(merged_statements))
    return new_rules

def resolution_process(knowledge_base, goal):
    pending_rules = list(knowledge_base)
    while pending_rules:
        current = pending_rules.pop(0)
        for existing in list(knowledge_base):
            if current != existing:
                new_generated = apply_resolution(current, existing)
                for new_rule in new_generated:
                    if new_rule not in knowledge_base:
                        pending_rules.append(new_rule)
                        knowledge_base.add(new_rule)
                    if not new_rule.statements:
                        return True
                    if goal in new_rule.statements:
                        return True
    return False

if __name__ == "__main__":
    kb = {
        Rule({Statement("P", ["a", "b"]), Statement("Q", ["a"])}),
        Rule({Statement("P", ["x", "y"])}),
        Rule({Statement("Q", ["y"]), Statement("R", ["y"])}),
        Rule({Statement("R", ["z"])}),
    }

    target = Statement("R", ["a"])
    found = resolution_process(kb, target)

    if found:
        print("Query is satisfiable.")
    else:
        print("Query is unsatisfiable.")
`;
const info9 = `
class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'

    def print_board(self):
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    def make_move(self, position):
        if self.board[position] == ' ':
            self.board[position] = self.current_player
            if self.check_winner(position):
                print(f"Player {self.current_player} wins!")
                return True
            elif ' ' not in self.board:
                print("It's a tie!")
                return True
            else:
                self.current_player = 'O' if self.current_player == 'X' else 'X'
                return False
        else:
            print("That position is already taken!")
            return False

    def check_winner(self, position):
        row_index = position // 3
        col_index = position % 3
        # Check row
        if all(self.board[row_index*3 + i] == self.current_player for i in range(3)):
            return True
        # Check column
        if all(self.board[col_index + i*3] == self.current_player for i in range(3)):
            return True
        # Check diagonal
        if row_index == col_index and all(self.board[i*3 + i] == self.current_player for i in range(3)):
            return True
        # Check anti-diagonal
        if row_index + col_index == 2 and all(self.board[i*3 + (2-i)] == self.current_player for i in range(3)):
            return True
        return False

def main():
    game = TicTacToe()
    while True:
        game.print_board()
        position = int(input(f"Player {game.current_player}, enter your position (0-8): "))
        if game.make_move(position):
            game.print_board()
            break

if __name__ == "__main__":
    main()
`;
app.get('/info1', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info1);   // return the Python code
});
app.get('/info2', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info2);   // return the Python code
});
app.get('/info3', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info3);   // return the Python code
});
app.get('/info4', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info4);   // return the Python code
});
app.get('/info5', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info5);   // return the Python code
});
app.get('/info6', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info6);   // return the Python code
});
app.get('/info7', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info7);   // return the Python code
});
app.get('/info8', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info8);   // return the Python code
});
app.get('/info9', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info9);   // return the Python code
});

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
module.exports = app;
