import math

def alpha_beta_pruning(state, depth, alpha, beta, maximizing_player):
    if depth == 0 or is_terminal_state(state):
        return evaluate_state(state)

    if maximizing_player:
        max_eval = -math.inf
        for move in get_possible_moves(state):
            eval = alpha_beta_pruning(move, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = math.inf
        for move in get_possible_moves(state):
            eval = alpha_beta_pruning(move, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval

def is_terminal_state(state):
    # Check for a win condition
    for row in state:
        if row.count(1) == 3:  # 'X' wins
            return True
        elif row.count(-1) == 3:  # 'O' wins
            return True
    for col in range(3):
        if state[0][col] == state[1][col] == state[2][col] == 1:  # 'X' wins
            return True
        elif state[0][col] == state[1][col] == state[2][col] == -1:  # 'O' wins
            return True
    if state[0][0] == state[1][1] == state[2][2] == 1 or state[0][2] == state[1][1] == state[2][0] == 1:  # 'X' wins
        return True
    elif state[0][0] == state[1][1] == state[2][2] == -1 or state[0][2] == state[1][1] == state[2][0] == -1:  # 'O' wins
        return True

    # Check for a draw condition
    for row in state:
        if 0 in row:  # If there is an empty cell, the game is not over
            return False
    return True  # If there are no empty cells, it's a draw

def get_possible_moves(state):
    possible_moves = []
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:  # If the cell is empty
                new_state = [row[:] for row in state]  # Create a copy of the current state
                new_state[i][j] = 1  # Assume 'X' makes the move
                possible_moves.append(new_state)
    return possible_moves

def evaluate_state(state):
    # Check for a win condition
    for row in state:
        if row.count(1) == 3:  # 'X' wins
            return 10
        elif row.count(-1) == 3:  # 'O' wins
            return -10
    for col in range(3):
        if state[0][col] == state[1][col] == state[2][col] == 1:  # 'X' wins
            return 10
        elif state[0][col] == state[1][col] == state[2][col] == -1:  # 'O' wins
            return -10
    if state[0][0] == state[1][1] == state[2][2] == 1 or state[0][2] == state[1][1] == state[2][0] == 1:  # 'X' wins
        return 10
    elif state[0][0] == state[1][1] == state[2][2] == -1 or state[0][2] == state[1][1] == state[2][0] == -1:  # 'O' wins
        return -10

    # If no winner, return a neutral score
    return 0

# Example usage:
initial_state = [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]]

# Assuming the game state is represented as a 2D list where 0 represents an empty cell,
# 1 represents a 'X' (cross), and -1 represents a 'O' (naught).

# Assuming 'is_terminal_state()' checks if the current state is a terminal state (win/loss/draw),
# and 'get_possible_moves()' generates possible moves from the current state.

# Evaluate the best move for the maximizing player (X) with alpha-beta pruning:
best_move = None
best_eval = -math.inf
alpha = -math.inf
beta = math.inf
for move in get_possible_moves(initial_state):
    eval = alpha_beta_pruning(move, depth=3, alpha=alpha, beta=beta, maximizing_player=True)
    if eval > best_eval:
        best_eval = eval
        best_move = move
    alpha = max(alpha, eval)

print("Best Move:", best_move)
print("Best Evaluation:", best_eval)
