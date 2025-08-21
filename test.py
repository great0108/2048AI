from game_numba import Env2048
from AI_2048 import find_best, evaluate
from utils import save_history
import numpy as np



# env = Env2048(4)
board = [
    [0, 2, 2, 4],
    [4, 0, 8, 32],
    [8, 16, 16, 64],
    [8, 32, 64, 128]
]
# board = [
#     [4, 4, 0, 0],
#     [4, 8, 32, 0],
#     [8, 32, 64, 2],
#     [8, 32, 64, 128]
# ]
board = np.array(board)
# evaluate(board)
move = find_best(board, 3)
print(move)