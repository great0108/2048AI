from game_numba import Env2048
from AI_2048 import find_best
from utils import save_history
from tqdm import tqdm

def generator():
  while True:
    yield

direction = ["left", "up", "right", "down"]
runs = 1
env = Env2048(4)
history = []
for i in range(runs):
    env.reset()
    canMove = [True] * 4
    for _ in tqdm(generator()):
        move = find_best(env.board, depth=3)
        history.append({'move': direction[move], 'board': env.board.copy()})
        board, score, over, canMove = env.step(move)
        if over:
            break

save_history(history, env.score)