import random
import sys
import numpy as np
from numba import njit

KEY_LEFT = 0
KEY_UP = 1
KEY_RIGHT = 2
KEY_DOWN = 3

@njit
def left(board):
    score = 0
    n = len(board)
    for i in range(n):
        last = board[i][0]
        fill = 1 if last else 0
        for j in range(1, n):
            if board[i][j]:
                x = board[i][j]
                board[i][j] = 0

                if last == x:
                    board[i][fill-1] = last*2
                    score += last*2
                    last = 0
                else:
                    last = x
                    board[i][fill] = x
                    fill += 1

    return score

@njit
def right(board):
    score = 0
    n = len(board)
    for i in range(n):
        last = board[i][n-1]
        fill = n-2 if last else n-1
        for j in range(n-2, -1, -1):
            if board[i][j]:
                x = board[i][j]
                board[i][j] = 0

                if last == x:
                    board[i][fill+1] = last*2
                    score += last*2
                    last = 0
                else:
                    last = x
                    board[i][fill] = x
                    fill -= 1

    return score

@njit
def up(board):
    score = 0
    n = len(board)
    for j in range(n):
        last = board[0][j]
        fill = 1 if last else 0
        for i in range(1, n):
            if board[i][j]:
                x = board[i][j]
                board[i][j] = 0

                if last == x:
                    board[fill-1][j] = last*2
                    score += last*2
                    last = 0
                else:
                    last = x
                    board[fill][j] = x
                    fill += 1

    return score

@njit
def down(board):
    score = 0
    n = len(board)
    for j in range(n):
        last = board[n-1][j]
        fill = n-2 if last else n-1
        for i in range(n-2, -1, -1):
            if board[i][j]:
                x = board[i][j]
                board[i][j] = 0

                if last == x:
                    board[fill+1][j] = last*2
                    score += last*2
                    last = 0
                else:
                    last = x
                    board[fill][j] = x
                    fill -= 1

    return score

@njit
def move(board, way):
    if way == KEY_LEFT:
        score = left(board)
    elif way == KEY_UP:
        score = up(board)
    elif way == KEY_RIGHT:
        score = right(board)
    else:
        score = down(board)
    return score

@njit
def able_move(board):
    moves = []
    for i in range(4):
        board2 = board.copy()
        move(board2, i)
        moves.append(not same_board(board, board2))
    return moves

@njit
def same_board(board1, board2):
    for i in range(4):
        for j in range(4):
            if board1[i][j] != board2[i][j]:
                return 0
    return 1

@njit
def is_end(board):
    canMov = able_move(board)
    over = True
    for i in range(len(canMov)):
        if canMov[i]:
            over = False
            break
    return over

@njit
def empty_cells(board):
    n = len(board)
    result = np.empty((n*n, 2), dtype=np.int64)
    idx = 0
    for i in range(n):
        for j in range(n):
            if board[i, j] == 0:
                result[idx, 0] = i
                result[idx, 1] = j
                idx += 1
    return result[:idx]

@njit
def random_tile(board):
    cells = empty_cells(board)
    if len(cells) == 0:
        return

    v = 2 if np.random.random() < 0.9 else 4
    idx = np.random.randint(len(cells))
    i, j = cells[idx]
    board[i][j] = v

@njit
def all_next_board(board):
    cells = empty_cells(board)
    num_cells = len(cells)
    if num_cells == 0:
        return
    
    board = np.expand_dims(board, axis=0) + np.zeros((num_cells*2, 4, 4))
    probs = np.zeros(num_cells*2)
    base_prob = 1 / num_cells
    for i in range(num_cells):
        a, b = cells[i]
        board[i*2][a][b] = 2
        board[i*2+1][a][b] = 4
        probs[i*2] = base_prob * 0.9
        probs[i*2+1] = base_prob * 0.1

    return board, probs

@njit
def next_2_board(board):
    cells = empty_cells(board)
    num_cells = len(cells)
    if num_cells == 0:
        return
    
    board = np.expand_dims(board, axis=0) + np.zeros((num_cells, 4, 4))
    probs = np.zeros(num_cells)
    base_prob = 1 / len(cells)
    for i in range(len(cells)):
        a, b = cells[i]
        board[i][a][b] = 2
        probs[i] = base_prob

    return board, probs

@njit
def step(board, way):
    last_board = board.copy()
    score = move(board, way)

    moved = not same_board(last_board, board)
    if moved:
        random_tile(board)

    canMov = able_move(board)
    over = True
    for i in range(len(canMov)):
        if canMov[i]:
            over = False
            break
    return score, over, canMov

@njit
def init_board(n):
    board = np.zeros((n, n))
    for _ in range(2):
        random_tile(board)
    return board

@njit
def board_hash(board):
    s = 0
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j]:
                s += 16 ** (i*4+j) * int(np.log2(board[i][j]))
    return s


class Env2048(object):
    def __init__(self, n):
        self.n = n
        self.score = 0
        self.board = init_board(n)

    def reset(self):
        self.score = 0
        self.board = init_board(self.n)
        return self.board

    def step(self, way):
        score, over, canMove = step(self.board, way)
        self.score += score
        return self.board, score, over, canMove

def test(runs=1):
    env = Env2048(4)
    for i in range(runs):
        env.reset()
        canMove = [True] * 4
        over = False
        while not over:
            move = random.choice([i for i in range(4) if canMove[i]])
            board, score, over, canMove = env.step(move)

if __name__ == '__main__':
    import timeit
    import sys
    print("testing first run")
    print(timeit.timeit('test()', globals=globals(), number=1))
    print("testing 10000 runs")
    print(timeit.timeit('test(10000)', globals=globals(), number=1))

    sys.exit(0)