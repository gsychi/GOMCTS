import numpy as np
from TTTEnvironment import TTTEnvironment


# w stands for # of wins, n stands for number of times node has been visited.
# N stands for number of times parent node is visited, and c is just an exploration constant that can be tuned.
# Q is the evaluation from 0 to 1 of the neural network
# L is a number of value 0 or 1. If the move is legal, then this value is 1. If the move is not, then this value is 0.
def UCT_Algorithm(w, n, c, N, q, L):
    # Provides a win rate score from 0 to 1
    selfPlayEvaluation = 0.5
    selfPlayEvaluation = np.divide(w, n, out=np.zeros_like(w), where=n!=0)
    nnEvaluation = q
    winRate = (nnEvaluation + selfPlayEvaluation) / 2

    # Exploration
    exploration = c * np.sqrt(np.log(N) / n)

    UCT = winRate + exploration
    return UCT * L


#UCT Algorithm used by Alpha Zero. Use this for now!!
def PUCT_Algorithm(w, n, c, N, q, L):
    # Provides a win rate score from 0 to 1
    selfPlayEvaluation = 0.5
    selfPlayEvaluation = np.divide(w, n, out=np.zeros_like(w), where=n != 0)
    nnEvaluation = q
    winRate = (nnEvaluation + selfPlayEvaluation) / 2

    # Exploration
    exploration = c * np.sqrt(N)/(1+n)

    UCT = winRate + exploration
    return UCT * L

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
        #self.gameStateSeen = np.zeros(9) Commented because it seems obsolete
        self.childrenStateSeen = np.zeros((1, 9))  # This is a 2D array
        self.childrenStateWin = np.zeros((1, 9))  # This is a 2D array
        self.childrenNNEvaluation = np.ones((1, 9))  # This is a 2D array

    def addToDictionary(self, position):
        self.dictionary[position]=len(self.dictionary)

        blank = np.zeros((1, 9))
        self.childrenStateSeen = np.concatenate((self.childrenStateSeen, blank), axis=0)
        self.childrenStateWin = np.concatenate((self.childrenStateWin, blank), axis=0)
        self.childrenNNEvaluation = np.concatenate((self.childrenNNEvaluation, np.random.rand(1,9)), axis=0)

    #choose argmax per (P)UCT Algorithm
    def chooseMove(self, position, legalMoves):
        index = self.dictionary[position]  # This returns a number based on the library

        # here we will assume that there is a legalMove function that works as followed:
        # If the first row of a tic tac toe row is all taken, then it returns [0,0,0,1,1,1,1,1,1]
        moveChoice = PUCT_Algorithm(self.childrenStateWin[index], self.childrenStateSeen[index], np.ones((1, 9))*2, np.sum(self.childrenStateSeen[index]), self.childrenNNEvaluation[index], legalMoves)

        return np.argmax(moveChoice)

    def initializePosition(self, pos): #adds pos to dictionary and concatenates new layers to MCTS arrays
        self.addToDictionary(pos)
        self.childrenStateSeen=np.concatenate((self.childrenStateSeen,np.zeros((1, 9))), axis=0)
        self.childrenStateWin = np.concatenate((self.childrenStateWin, np.zeros((1, 9))), axis=0)
        self.childrenNNEvaluation = np.concatenate((self.childrenNNEvaluation, np.zeros((1, 9))), axis=0)

    def simulation(self, originalPosition):

        # We store the information in the simulation in a temporary array, before adding everything to the database.
        OstatesExplored = np.zeros((1, 1))  # Store it as an array of directories for the seen stuff
        OactionsDone = np.zeros((1, 9))
        XstatesExplored = np.zeros((1, 1))  # Store it as an array of directories for the seen stuff
        XactionsDone = np.zeros((1, 9))

        # initialize the simulation and set the state of the board.
        tempBoard = TTTEnvironment()
        tempBoard.state = TTTEnvironment.stringToState(tempBoard, originalPosition)
        tempBoard.setValues()
        end = tempBoard.check_Win()

        #while loop should start here
        while end == 2:
            # Retrieve string representation of the current position.
            currentPosition = tempBoard.stateToString()
            # Make sure that new positions are added to the database
            if currentPosition not in self.dictionary:
                self.addToDictionary(currentPosition)

            #Now, you can select a move.
            index = self.chooseMove(currentPosition, tempBoard.legalMove())
            tempBoard.makeMove(index)
            # switch turn. (+1%2 is right but it isn't working at the moment)
            if str(tempBoard.turn) == '1' or tempBoard.turn == 1:
                tempBoard.turn = 0
            else:
                tempBoard.turn = 1
            tempBoard.updateState()
            print("STATE: " + tempBoard.stateToString())
            print(tempBoard.Xstate)
            print(tempBoard.Ostate)
            end = tempBoard.check_Win()

        return end

x = MonteCarlo()

print(x.simulation('0000000000000000000'))
print(x.dictionary)

"""
#x.simulation('0000000000000000000')
w = np.zeros((1,9))
n = np.zeros((1,9))
c = np.ones((1,9))
q = np.random.rand(1,9)
L = np.ones((1,9))
print(PUCT_Algorithm(w, n, c, np.sum(n), q, L))

print(x.chooseMove('0000000000000000000', L))

"""
