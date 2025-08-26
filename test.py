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

board = [[ 4,  8,  2, 32],
 [ 2,  2,  8, 16],
 [ 0,  0,  2,  2],
 [ 0,  0,  8,  0]]

# board = [[16,  4,  0,  0],
# [64, 16,  2, 2],
# [256, 128,   8,   2],
# [512,   2,  16,   4]]


board = np.array(board)
print(evaluate(board))
for i in range(4):
    print(expectimax_pvs(board, i+1, -1e+6))


def test(runs=1):
    for i in range(runs):
        move = find_best(board)
        print(move)

if __name__ == '__main__':
    import timeit
    import sys
    # print("testing first run")
    # print(timeit.timeit('test()', globals=globals(), number=1))
    # print("testing 10 runs")
    # print(timeit.timeit('test(10)', globals=globals(), number=1))

    sys.exit(0)
