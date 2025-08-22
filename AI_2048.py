import numpy as np
from game_numba import *
from numba import njit, types
from numba.typed import Dict

@njit
def find_best(board, depth=0):
    if depth == 0:
        depth = max(3, (16 - len(empty_cells(board))) // 3)

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
        #     print(next_boards[i], v)

        # print(mov, value)

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
            v = expectimax_pvs(next_boards[i], depth-1, alpha)
            value += v * probs[i]

        if value > best:
            best = value
        if best > alpha:
            alpha = best

    return best

@njit
def pre_evaluate():
    value_table = np.zeros(2**16)
    for i in range(2 ** 16):
        line = [
            (i >> 0) % 16,
            (i >> 4) % 16,
            (i >> 8) % 16,
            (i >> 12) % 16
        ]
        value_table[i] = evaluate_line(line)
    return value_table

@njit
def evaluate_line(line):
    sum_power = 3.5
    sum_weight = 11
    monotonic_power = 4
    monotonic_weight = 20
    merge_weight = 1400
    empty_weight = 270

    sum_value = 0
    empty = 0
    merges = 0
    monotonic = 0

    monotonic_left = 0
    monotonic_right = 0
    prev = 0

    for i in range(4):
        rank = line[i]
        sum_value += rank ** sum_power
        if rank == 0:
            empty += 1
        else:
            if prev == rank:
                merges += 1
            prev = rank

        if i > 0:
            prev_rank = line[i]
            if rank > prev_rank:
                monotonic_left += prev_rank ** monotonic_power - rank ** monotonic_power
            else:
                monotonic_right += rank ** monotonic_power - prev_rank ** monotonic_power

    monotonic += max(monotonic_left, monotonic_right)

    value = empty * empty_weight + merges * merge_weight \
        + monotonic * monotonic_weight + sum_value * sum_weight

    return value

@njit   
def evaluate(board):
    rank = np.zeros((4, 4), dtype=types.int32)
    for i in range(4):
        for j in range(4):
            rank[i][j] = 0 if board[i][j] == 0 else int(np.log2(board[i][j]))

    rank2 = rank.T
    return _evaluate(rank) + _evaluate(rank2)

@njit
def _evaluate(rank):
    global value_table
    value = 0
    for i in range(4):
        idx = rank[i][0] + rank[i][1] * 2**4 + rank[i][2] * 2**8 + rank[i][3] * 2**12

        value += value_table[idx]

    return value

value_table = pre_evaluate()