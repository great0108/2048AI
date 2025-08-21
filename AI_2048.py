import numpy as np
from game_numba import *
from numba import njit, types
from numba.typed import Dict

@njit
def find_best(board, depth):
    best_move = 3
    best = -10**9
    alpha = best
    for mov in [3, 0, 2, 1]: # prefer down
        new_board = board.copy()
        move(new_board, mov)
        if np.array_equal(new_board, board):
            continue

        value = 0
        next_boards, probs = all_next_board(new_board)
        for i in range(len(next_boards)):
            v = expectimax_pvs(next_boards[i], depth-1, alpha)
            value += v * probs[i]
            print(next_boards[i], v)

        print(mov, value)

        if value > best:
            best = value
            best_move = mov
        if best > alpha:
            alpha = best

    return best_move

@njit
def expectimax_pvs(board, depth, alpha):
    if is_end(board):
        return -200000

    if depth == 0:
        return evaluate(board)

    best = -10**9
    for mov in [3, 0, 2, 1]: # prefer down
        new_board = board.copy()
        move(new_board, mov)
        if np.array_equal(new_board, board):
            continue

        value = 0
        next_boards, probs = all_next_board(new_board)
        for i in range(len(next_boards)):
            value += expectimax_pvs(next_boards[i], depth-1, alpha) * probs[i]

        if value > best:
            best = value
        if best > alpha:
            alpha = best

    return best

@njit   
def evaluate(board):
    sum_power = 3.5
    sum_weight = 11
    monotonic_power = 4
    monotonic_weight = 5
    merge_weight = 350
    empty_weight = 270

    sum_value = 0
    empty = 0
    merges = 0
    monotonic = 0

    # row wise
    for i in range(4):
        monotonic_left = 0
        monotonic_right = 0
        prev = 0
        counter = 0

        for j in range(4):
            rank = 0 if board[i][j] == 0 else np.log2(board[i][j])
            sum_value += rank ** sum_power
            if rank == 0:
                empty += 1
            else:
                if prev == rank:
                    counter += 1
                elif counter > 0:
                    merges += 1 + counter
                    counter = 0
                prev = rank

            if j > 0:
                prev_rank = 0 if board[i][j-1] == 0 else np.log2(board[i][j-1])
                if rank > prev_rank:
                    monotonic_left += prev_rank ** monotonic_power - rank ** monotonic_power
                else:
                    monotonic_right += rank ** monotonic_power - prev_rank ** monotonic_power

        if counter > 0:
            merges += 1 + counter
        monotonic += max(monotonic_left, monotonic_right)

    # column wise
    for i in range(4):
        monotonic_up = 0
        monotonic_down = 0
        prev = 0
        counter = 0

        for j in range(4):
            rank = 0 if board[j][i] == 0 else np.log2(board[j][i])
            if rank:
                if prev == rank:
                    counter += 1
                elif counter > 0:
                    merges += 1 + counter
                    counter = 0
                prev = rank

            if j > 0:
                prev_rank = 0 if board[j-1][i] == 0 else np.log2(board[j-1][i])
                if rank > prev_rank:
                    monotonic_up += prev_rank ** monotonic_power - rank ** monotonic_power
                else:
                    monotonic_down += rank ** monotonic_power - prev_rank ** monotonic_power

        if counter > 0:
            merges += 1 + counter
        monotonic += max(monotonic_up, monotonic_down)

    value = empty * empty_weight + merges * merge_weight \
        + monotonic * monotonic_weight + sum_value * sum_weight

    return value