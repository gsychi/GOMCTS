import numpy as np
import TTTEnvironment

class MonteCarlo():

    # We will use three lists:
    # seenStates stores the gameStates that have been seen. This is a library.
    # parentSeen stores the times that the current gameState has been seen
    # Each game state corresponds to arrays of the possible moves (<=9)
    # There are 3 points information stored for each of the children
    # - win count, number of times visited, and neural network evaluation
    # This is helpful because we get to use numpy stuffs.

    def __init__(self):
        # This is a dictionary. Information will be updated as playouts start. ['stateToString': 0]
        # 0 corresponds to position 0 on all the other arrays, 1 corresponds to position 1...hmmm.
        self.dictionary = {
            '0000000000000000000': 0  # empty board corresponds to position 0 on numpy arrays
        }
        self.gameStateSeen = np.zeros(9)
        self.childrenStateSeen = np.zeros((1, 9))  # This is a 2D array
        self.childrenStateWin = np.zeros((1, 9))  # This is a 2D array
        self.childrenNNEvaluation = np.zeros((1, 9))  # This is a 2D array

    def simulation(self):
        # We store the information in the simulation in a temporary array, before adding everything to the database.
        turns = 0

        OstatesExplored = np.zeros(1)  # Store it as an array of directories for the seen stuff
        OactionsDone = np.zeros((1,9))
        XstatesExplored = np.zeros(1)  # Store it as an array of directories for the seen stuff
        XactionsDone = np.zeros((1,9))

        #some tic tac toe initializing shit here
        # foo = boardToNumber for now
        position = '0000000000000000000'
        
        board=TTTEnvironment()
        
        board.state=TTTEnvironment.stringToState(position)
        TTTEnvironment.setValues(board)
        
        legalMoves=board.legalMove(self)

        #if game state is seen before,
        if position in self.dictionary:
            move = self.chooseMove(self.chooseMove(position, legalMoves))
            if (turns%2)==0:
                #change tic tac toe board
            if (turns % 2) == 1:
                # change tic tac toe board

    #choose argmax per (P)UCT Algorithm
    def chooseMove(self, position, legalMoves):
        index = self.dictionary[position]  # This returns a number based on the library

        # here we will assume that there is a legalMove function that works as followed:
        # If the first row of a tic tac toe row is all taken, then it returns [0,0,0,1,1,1,1,1,1]
        moveChoice = UCT_Algorithm(self.childrenStateWin[index], self.childrenStateSeen[index], 2, self.gameStateSeen[index], self.childrenNNEvaluation[index], legalMoves)
        return np.argmax(moveChoice)

# w stands for # of wins, n stands for number of times node has been visited.
# N stands for number of times parent node is visited, and c is just an exploration constant that can be tuned.
# Q is the evaluation from 0 to 1 of the neural network
# L is a number of value 0 or 1. If the move is legal, then this value is 1. If the move is not, then this value is 0.
def UCT_Algorithm(w, n, c, N, q, L):
    # Provides a win rate score from 0 to 1
    selfPlayEvaluation = 0
    if n != 0:
        selfPlayEvaluation = w / n
    nnEvaluation = q
    winRate = (nnEvaluation + selfPlayEvaluation) / 2

    # Exploration
    exploration = c * np.sqrt(np.log(N) / n)

    UCT = winRate + exploration
    return UCT * L

#UCT Algorithm used by Alpha Zero
def PUCT_Algorithm(w, n, c, N, q, L):
    # Provides a win rate score from 0 to 1
    selfPlayEvaluation = 0
    if n != 0:
        selfPlayEvaluation = w / n
    nnEvaluation = q
    winRate = (nnEvaluation + selfPlayEvaluation) / 2

    # Exploration
    exploration = c * np.sqrt(N)/(1+n)

    UCT = winRate + exploration
    return UCT * L



x = MonteCarlo()
print(x.gameStates)
