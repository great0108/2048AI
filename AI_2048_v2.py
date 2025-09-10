import numpy as np
from simulate_2048_v2 import Batch2048EnvSimulator
from tqdm import tqdm
import ray

def find_best(board, depth=None, use_ray=False):
    if depth == None:
        depth = 2

    if len(board.shape) == 1:
        board = np.expand_dims(board, axis=0)

    if Batch2048EnvSimulator.able_move(board).sum() == 0:
        return 255

    moved_boards, index, move = Batch2048EnvSimulator.all_move_boards(board)
    next_boards, index2, num_cases = Batch2048EnvSimulator.all_next_boards(moved_boards)
    
    if depth == 0:
        value = evaluate(next_boards)

    else:
        if use_ray:
            obj = []
            for i in range(len(next_boards)):
                obj.append(expectimax_ray.remote(next_boards[i:i+1], depth-1))
            value = np.array(ray.get(obj)).flatten()
            
        else:
            value = expectimax(next_boards, depth-1)

    value[0::2] *= 0.9
    value[1::2] *= 0.1
    value = np.bincount(index2, value) / num_cases
    return move[np.argmax(value)]

def expectimax(boards, depth):
    moved_boards, index, move = Batch2048EnvSimulator.all_move_boards(boards)
    if moved_boards.shape[0] == 0:
        return np.full((boards.shape[0],), -1e6, dtype=np.float32)
    
    if depth == 0:
        next_boards, index2, num_cases = Batch2048EnvSimulator.all_next_boards(moved_boards, only_2=True)
        value = evaluate(next_boards)

    else:
        next_boards, index2, num_cases = Batch2048EnvSimulator.all_next_boards(moved_boards)
        value = expectimax(next_boards, depth-1)
        value[0::2] *= 0.9
        value[1::2] *= 0.1
    
    value = np.bincount(index2, value) / num_cases
    n = boards.shape[0]
    out = np.full(n, -1e6, dtype=np.float32)
    np.maximum.at(out, index, value)
    return out

expectimax_ray = ray.remote(expectimax)

def pre_evaluate():
    value_table = np.zeros((4, 2**16), dtype=np.float32)
    locate_weight = np.array([[0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [2.0, 3.0, 4.0, 5.0],
                    [8.0, 6.0, 5.0, 4.0]], dtype=np.float32)
    for i in range(2 ** 16):
        line = [
            (i >> 0) % 16,
            (i >> 4) % 16,
            (i >> 8) % 16,
            (i >> 12) % 16
        ]
        value_table[0, i] = evaluate_line(line, locate_weight[0])
        value_table[1, i] = evaluate_line(line, locate_weight[1])
        value_table[2, i] = evaluate_line(line, locate_weight[2])
        value_table[3, i] = evaluate_line(line, locate_weight[3])
    return value_table

def evaluate_line(line, locate_weight):
    # sum_power = 3.5
    # sum_weight = 11
    locate_power = 3
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
        sum_value += rank ** locate_power * locate_weight[i]
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

def evaluate(board):
    trans_board = board.copy()
    Batch2048EnvSimulator._transpose_inplace(trans_board)  
    return _evaluate(board) + _evaluate(trans_board, locate=False)

def _evaluate(rank, locate=True):
    if locate:
        return (value_table[0][rank[:, 0]] +
        value_table[1][rank[:, 1]] +
        value_table[2][rank[:, 2]] +
        value_table[3][rank[:, 3]])
    
    v_table = value_table[0]
    return v_table[rank].sum(axis=1)

value_table = pre_evaluate()

if __name__ == "__main__":
    num_env = 1
    boards = Batch2048EnvSimulator.init_board(num_env)
    for i in tqdm(range(1000)):
        move = find_best(boards, depth=3, use_ray=True)

    # move = find_best(boards, depth=3)
    # print(boards, move)