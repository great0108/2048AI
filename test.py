from game_numba import Env2048
from AI_2048 import find_best, evaluate, expectimax_pvs
from utils import save_history
import numpy as np



# env = Env2048(4)
board = [
    [0, 0, 2, 16],
    [2, 4, 8, 16],
    [2, 4, 2, 16],
    [0, 2, 8, 2]
]

# board = [[0, 0, 0, 0],
# [0, 0, 0, 4],
# [2, 4, 8, 0],
# [256, 128, 128, 0]]


board = [[0, 0, 0, 0],
[0, 0, 2, 2],
[2, 4, 4, 4],
[256, 128,  64,  64]]

board = np.array(board)
print(find_best(board, 3))
# for i in range(4):
#     print(find_best(board, i+1))

# def test(runs=1):
#     for i in range(runs):
#         move = find_best(board)

a = np.uint64(2**32 + 16)
b = np.zeros((1, 4), dtype=np.uint16)
b[0] = a
print(b)

if __name__ == '__main__':
    import timeit
    import sys
    # print("testing first run")
    # print(timeit.timeit('test()', globals=globals(), number=1))
    # print("testing 100 runs")
    # print(timeit.timeit('test(100)', globals=globals(), number=1))

    sys.exit(0)
