import numpy as np

class MonteCarlo():

    # We will use three lists:
    # seenStates stores the gameStates that have been seen. This is a library.
    # parentSeen stores the times that the current gameState has been seen
    # Each game state corresponds to arrays of the possible moves (<=9)
    # There are 3 points information stored for each of the children - win count, number of times visited, and neural network evaluation
    # This is helpful because we get to use numpy stuffs.

    def __init__(self):


        # This is a dictionary. Information will be updated as playouts start. ['stateToString': 0]
        # 0 corresponds to position 0 on all the other arrays, 1 corresponds to position 1...hmmm.
        self.gameStates = {}

        self.childrenStatesSeen = np.zeros((1, 9)) #This is a 2D array
        self.childrenStatesWin = np.zeros((1, 9)) #This is a 2D array
        self.childrenNNEvaluation = np.zeros((1, 9)) #This is a 2D array

    def simulation(self):
        # We store the information in the simulation in a temporary array, before adding everything to the database.

        statesExplored = np.zeros((1)) #Store it as an array of seen stuff
        actionsDone = np.zeros((1))

        #here we will assume that there is a legalMove function that works as followed:
        #If the first row of a tic tac toe row is all taken, then it returns [0,0,0,1,1,1,1,1,1]

    def chooseMove(self, ):


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
