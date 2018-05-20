import numpy as np

class GoEnvironment:

    def __init__(self):
        self.board = np.zeros((9,9,2))
        self.plies_before_board = np.zeros((3,9,9,2))
        self.turns = 0
        self.alphabet = 'ABCDEFGHI'
        
