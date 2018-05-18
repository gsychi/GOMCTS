import numpy as np

# in this code, a list will be used to store each position.
# [0,0,0,0,0,0,0,0,0]
# -> this corresponds to 2 arrays for wins and number of times visited]
# -> then each position has a value for the # of times it is visited = parent node N value

class MonteCarlo():

    # We will use three lists:
    # seenStates stores the gameStates that have been seen. This is a library.
    # parentSeen stores the times that the current gameState has been seen
    # Each game state corresponds to arrays of the possible moves (<=9)
    # There are 3 points information stored for each of the children - win count, number of times visited, and neural network evaluation
    # This is helpful because we get to use numpy stuffs.

    def __init__(self):
        self.gameStates = {} # This is a dictionary. Information will be updated as playouts start. [
        self.childrenStatesSeen = np.zeros((1, 9)) #This is a 2D array
        self.childrenStatesWin = np.zeros((1, 9)) #This is a 2D array
        self.childrenNNEvaluation = np.zeros((1, 9)) #This is a 2D array


# w stands for # of wins, n stands for number of times node has been visited.
# N stands for number of times parent node is visited, and c is just an exploration constant that can be tuned.
# Q is the evaluation from 0 to 1 of the neural network=
def UCT_Algorithm(w, n, c, N, q):
    # Provides a win rate score from 0 to 1
    selfPlayEvaluation = 0
    if n != 0:
        selfPlayEvaluation = w / n
    nnEvaluation = q
    winRate = (nnEvaluation + selfPlayEvaluation) / 2

    # Exploration
    exploration = c * math.sqrt(math.log(N) / n)

    UCT = winRate + exploration
    return UCT


x = MonteCarlo()
print(x.gameStates)
