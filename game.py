import random
import sys
import numpy as np

KEY_LEFT = 0
KEY_UP = 1
KEY_RIGHT = 2
KEY_DOWN = 3

class Env2048(object):
    def __init__(self, n):
        self.n = n
        self.reset()

    def reset(self):
        self.score = 0
        self.board = [[0] * self.n for i in range(self.n)]
        self.randomTile()
        self.randomTile()
        return self.board

    def step(self, way):
        last_board = [row[:] for row in self.board]
        score = self.move(self.board, way)
        self.score += score

        moved = (last_board != self.board)
        if moved:
            self.randomTile()
   
        canMove = self.canMove()
        over = not any(canMove)
        return self.board, score, over, canMove

    def move(self, board, way):
        if way == KEY_LEFT:
            score = self.left(board)
        elif way == KEY_UP:
            score = self.up(board)
        elif way == KEY_RIGHT:
            score = self.right(board)
        else:
            score = self.down(board)
        return score

    def left(self, board):
        score = 0
        n = self.n
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

    def right(self, board):
        score = 0
        n = self.n
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

    def up(self, board):
        score = 0
        n = self.n
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

    def down(self, board):
        score = 0
        n = self.n
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

    def canMove(self):
        move = []
        for i in range(4):
            board = [row[:] for row in self.board]
            self.move(board, i)
            move.append(self.board != board)
        return move

    def isEnd(self):
        return not any(self.canMove())

    def emptyCells(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.board[i][j] == 0:
                    yield i, j

    def randomTile(self):
        cells = list(self.emptyCells())
        if not cells:
            return False

        if random.random() < 0.9:
            v = 2
        else:
            v = 4

        cid = random.choice(cells)
        self.board[cid[0]][cid[1]] = v
        return True

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