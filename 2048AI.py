import numpy as np

def find_best_move(board):
    best_move = 0
    best_score = 0
    for move in range(4):
        score = score_move(board, move)

        if(score > best_score):
            best_score = score
            best_move = move
    return best_move

def score_move(board, move, depth=0):
    depth