import random
import sys
import numpy as np

KEY_LEFT = 1
KEY_UP = 2
KEY_RIGHT = 3
KEY_DOWN = 4

class Board(object):
    def __init__(self):
        self.board = [[0] * 4 for i in range(4)]
        self.randomTile()
        self.randomTile()

        self.score = 0

    def move(self, way):
        last_board = [row[:] for row in self.board]
        if way == KEY_LEFT:
            self.left()
        elif way == KEY_UP:
            self.up()
        elif way == KEY_RIGHT:
            self.right()
        else:
            self.down()

        moved = (last_board != self.board)
        if moved:
            self.randomTile()
   
        canMove = self.canMove()
        over = not any(canMove)
        return self.board, over, canMove

    def left(self):
        for i in range(4):
            last = self.board[i][0]
            fill = 1 if last else 0
            for j in range(1, 4):
                if self.board[i][j]:
                    x = self.board[i][j]
                    self.board[i][j] = 0

                    if last == x:
                        self.board[i][fill-1] = last*2
                        last = 0
                    else:
                        last = x
                        self.board[i][fill] = x
                        fill += 1

    def right(self):
        for i in range(4):
            last = self.board[i][3]
            fill = 2 if last else 3
            for j in range(2, -1, -1):
                if self.board[i][j]:
                    x = self.board[i][j]
                    self.board[i][j] = 0

                    if last == x:
                        self.board[i][fill+1] = last*2
                        last = 0
                    else:
                        last = x
                        self.board[i][fill] = x
                        fill -= 1

    def up(self):
        for j in range(4):
            last = self.board[0][j]
            fill = 1 if last else 0
            for i in range(1, 4):
                if self.board[i][j]:
                    x = self.board[i][j]
                    self.board[i][j] = 0

                    if last == x:
                        self.board[fill-1][j] = last*2
                        last = 0
                    else:
                        last = x
                        self.board[fill][j] = x
                        fill += 1

    def down(self):
        for j in range(4):
            last = self.board[3][j]
            fill = 2 if last else 3
            for i in range(2, -1, -1):
                if self.board[i][j]:
                    x = self.board[i][j]
                    self.board[i][j] = 0

                    if last == x:
                        self.board[fill+1][j] = last*2
                        last = 0
                    else:
                        last = x
                        self.board[fill][j] = x
                        fill -= 1

    def canMove(self):
        cells = list(self.emptyCells())
        if cells:
            return [True, True, True, True]

        row = self.canMoveRow()
        column = self.canMoveColumn()
        return [row, column, row, column]

    def canMoveRow(self):
        for i in range(4):
            for j in range(3):
                if self.board[i][j] == self.board[i][j+1]:
                    return True

        return False

    def canMoveColumn(self):
        for j in range(4):
            for i in range(3):
                if self.board[i][j] == self.board[i+1][j]:
                    return True

        return False

    def isEnd(self):
        return not any(self.canMove())

    def emptyCells(self):
        for i in range(4):
            for j in range(4):
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

def test():
    runs = 1
    for size in (4,):
        scores = np.zeros(runs)
        for i in range(runs):
            b = Board()
            canMove = [True] * 4
            over = False
            while not over:
                move = [i+1 for i in range(4) if canMove[i]]
                board, over, canMove = b.move(random.randint(1,4))
            # scores[i] = b.score

if __name__ == '__main__':
    

    import timeit
    import sys
    print("testing first run")
    print(timeit.timeit('test()', globals=globals(), number=1))
    print("testing 10000 runs")
    print(timeit.timeit('test()', globals=globals(), number=10000))


    sys.exit(0)