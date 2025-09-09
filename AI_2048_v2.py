import numpy as np
from simulate_2048_v2 import Batch2048EnvSimulator

def find_best(board, depth=0):
    if depth == 0:
        depth = 5

    best_move = 3
    best = -1e+6
    alpha = best
    first = True
    for mov in [3, 0, 2, 1]: # prefer down
        new_board = copy_board(board)
        move(new_board, mov)
        if np.array_equal(new_board, board):
            continue

        value = 0
        next_boards, probs = all_next_board(new_board)
        # if not first and depth > 1:
        #     for i in range(len(next_boards)):
        #         v = evaluate(next_boards[i])
        #         value += v * probs[i]

        # if value < alpha:
        #     continue

        value = 0
        for i in range(len(next_boards)):
            v = expectimax_pvs(next_boards[i], depth-1, alpha)
            value += v * probs[i]
        #     print(next_boards[i], v)

        # print(mov, value)

        if value > best:
            best = value
            best_move = mov
            alpha = min(value - 5000, value * 1.2, value * 0.8)

    if best == -1e+6:
        print("ai cannot choose move")
        for mov in [3, 0, 2, 1]: # prefer down
            new_board = copy_board(board)
            move(new_board, mov)
            if not np.array_equal(new_board, board):
                return mov

    return best_move

def expectimax_pvs(board, depth, alpha):
    if depth == 0:
        return evaluate(board)

    best = -1e+6
    first = True
    for mov in [3, 0, 2, 1]: # prefer down
        new_board = copy_board(board)
        move(new_board, mov)
        if same_board(new_board, board):
            continue

        value = 0
        if depth < 3:
            next_boards, probs = next_2_board(new_board)
        else:
            next_boards, probs = all_next_board(new_board)
        if not first and depth > 1:
            for i in range(len(next_boards)):
                v = evaluate(next_boards[i])
                value += v * probs[i]

        if value < alpha:
            continue

        value = 0
        first = False
        for i in range(len(next_boards)):
            v = expectimax_pvs(next_boards[i], depth-1, alpha)
            value += v * probs[i]

        if value > best:
            best = value
            alpha = max(alpha, min(value - 5000, value * 1.2, value * 0.8))

    return best

def pre_evaluate():
    value_table = np.zeros(2**16, dtype=np.float64)
    for i in range(2 ** 16):
        line = [
            (i >> 0) % 16,
            (i >> 4) % 16,
            (i >> 8) % 16,
            (i >> 12) % 16
        ]
        value_table[i] = evaluate_line(line)
    return value_table

def evaluate_line(line):
    # sum_power = 3.5
    # sum_weight = 11
    monotonic_power = 3
    monotonic_weight = 10
    merge_weight = 40
    empty_weight = 27

    sum_value = 0
    empty = 0
    merges = 0

    monotonic_left = 0
    monotonic_right = 0
    prev = 0

    for i in range(4):
        rank = line[i]
        sum_value += 2 ** rank
        if rank == 0:
            empty += 1
        else:
            if prev == rank:
                merges += rank
            prev = rank

        if i > 0:
            prev_rank = line[i-1]
            if rank > prev_rank:
                monotonic_left += prev_rank ** monotonic_power - rank ** monotonic_power
            else:
                monotonic_right += rank ** monotonic_power - prev_rank ** monotonic_power

    monotonic = max(monotonic_left, monotonic_right)

    value = empty * empty_weight + merges * merge_weight \
        + monotonic * monotonic_weight + sum_value

    return value

snake = np.array([[0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [2.0, 3.0, 4.0, 5.0],
                    [8.0, 6.0, 5.0, 4.0]], dtype=np.float64)
snake = 2 ** snake

def evaluate(board):
    value = 0.0
    rank = np.zeros((4, 4), dtype=np.int32)
    for i in range(4):
        for j in range(4):
            if board[i][j] != 0:
                rank[i][j] = int(np.log2(board[i][j]))
                value += (rank[i][j] ** 3) * snake[i][j]

    rank2 = rank.T
    return _evaluate(rank) + _evaluate(rank2) + value / 50

def _evaluate(rank):
    val = 0.0
    for i in range(4):
        idx = (rank[i, 0]
               + (rank[i, 1] << 4)
               + (rank[i, 2] << 8)
               + (rank[i, 3] << 12))
        val += value_table[idx]
    return val

value_table = pre_evaluate()