import numpy as np
from simulate_2048_v2 import Batch2048EnvSimulator
from tqdm import tqdm
import ray
from ray.experimental import tqdm_ray
import time

@ray.remote(num_cpus=2)
def expectimax_remote(boards, depth):
    return AI_2048.expectimax2(boards, depth+1)

class AI_2048():
    value_table = None

    @staticmethod
    def init(params=None):
        if params is None:
            params = {
                "sum_weight" : 660,
                "sum_power" : 3.5,
                "locate_power" : 0,
                "locate_weight" : np.array([[0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [2.0, 3.0, 4.0, 5.0],
                        [8.0, 6.0, 5.0, 4.0]], dtype=np.float32),
                "monotonic_power" : 3.7,
                "monotonic_weight" : 28,
                "merge_weight" : 750,
                "empty_weight" : 300
            }

        AI_2048.value_table = AI_2048.pre_evaluate(params)

    @staticmethod
    def find_best(board, depth=None, use_ray=False):

        if len(board.shape) == 1:
            board = np.expand_dims(board, axis=0)

        if depth == None:
            depth = max(2, (12 - Batch2048EnvSimulator.empty_space_nums(board)[0]) // 3)

        if Batch2048EnvSimulator.able_move(board).sum() == 0:
            return 255

        moved_boards, index, move = Batch2048EnvSimulator.all_move_boards(board)
        next_boards, index2, num_cases = Batch2048EnvSimulator.all_next_boards(moved_boards)
        
        if depth == 0:
            value = AI_2048.evaluate(next_boards)

        else:
            if use_ray:
                obj = []
                for i in range(len(next_boards)):
                    obj.append(expectimax_remote.remote(np.array(next_boards[i:i+1]), depth-1))
                value = np.array(ray.get(obj)).flatten()
                
            else:
                value = AI_2048.expectimax(next_boards, depth-1)

        value[0::2] *= 0.9
        value[1::2] *= 0.1
        value = np.bincount(index2, value) / num_cases
        return move[np.argmax(value)]

    @staticmethod
    def expectimax(boards, depth):
        moved_boards, index, move = Batch2048EnvSimulator.all_move_boards(boards)
        if moved_boards.shape[0] == 0:
            return np.full((boards.shape[0],), -1e6, dtype=np.float32)
        
        if depth == 0:
            next_boards, index2, num_cases = Batch2048EnvSimulator.all_next_boards(moved_boards, only_2=True)
            value = AI_2048.evaluate(next_boards)

        else:
            next_boards, index2, num_cases = Batch2048EnvSimulator.all_next_boards(moved_boards)
            value = AI_2048.expectimax(next_boards, depth-1)
            value[0::2] *= 0.9
            value[1::2] *= 0.1
        
        value = np.bincount(index2, value) / num_cases
        n = boards.shape[0]
        out = np.full(n, -1e6, dtype=np.float32)
        np.maximum.at(out, index, value)
        return out

    @staticmethod
    def pre_evaluate(params):
        value_table = np.zeros((4, 2**16), dtype=np.float32)
        locate_weight = params["locate_weight"]
        for i in range(2 ** 16):
            line = [
                (i >> 0) % 16,
                (i >> 4) % 16,
                (i >> 8) % 16,
                (i >> 12) % 16
            ]
            value_table[0, i] = AI_2048.evaluate_line(line, locate_weight[0], params)
            value_table[1, i] = AI_2048.evaluate_line(line, locate_weight[1], params)
            value_table[2, i] = AI_2048.evaluate_line(line, locate_weight[2], params)
            value_table[3, i] = AI_2048.evaluate_line(line, locate_weight[3], params)
        return value_table

    @staticmethod
    def evaluate_line(line, locate_weight, params):
        # sum_power = 3.5
        sum_weight = params["sum_weight"]
        locate_power = params["locate_power"]
        monotonic_power = params["monotonic_power"]
        monotonic_weight = params["monotonic_weight"]
        merge_weight = params["merge_weight"]
        empty_weight = params["empty_weight"]

        sum_value = 0
        locate_value = 0
        empty = 0
        merges = 0

        monotonic_left = 0
        monotonic_right = 0
        prev = 0

        for i in range(4):
            rank = line[i]
            locate_value += 2 ** rank * locate_weight[i]
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
            + monotonic * monotonic_weight + sum_value * sum_weight + locate_value

        return value

    @staticmethod
    def evaluate(board):
        trans_board = board.copy()
        Batch2048EnvSimulator._transpose_inplace(trans_board)  
        return AI_2048._evaluate(board) + AI_2048._evaluate(trans_board, locate=False)

    @staticmethod
    def _evaluate(rank, locate=True):
        value_table = AI_2048.value_table
        if locate:
            return (value_table[0][rank[:, 0]] +
            value_table[1][rank[:, 1]] +
            value_table[2][rank[:, 2]] +
            value_table[3][rank[:, 3]])
        
        v_table = value_table[0]
        return v_table[rank].sum(axis=1)

@ray.remote
def test(steps, bar):
    boards = Batch2048EnvSimulator.init_board(1)
    for step in range(steps):
        move = AI_2048.find_best(boards, depth=3)
        bar.update.remote(1)

if __name__ == "__main__":
    num_env = 1
    ray.init(num_cpus=32)
    AI_2048.init()
    Batch2048EnvSimulator.init()
    boards = Batch2048EnvSimulator.init_board(num_env)

    # for i in tqdm(range(1000)):
    #     move = AI_2048.find_best(boards, depth=3)

    # move = find_best(boards, depth=3)
    # print(boards, move)

    # with multi processing
    n = 1000
    split = 20  # cpu core num

    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    bar = remote_tqdm.remote(total=n)
    obj = []
    for i in range(split):
        step = n // split if i != 0 else n // split + n % split
        obj.append(test.remote(step, bar))

    ray.get(obj)
    bar.close.remote()