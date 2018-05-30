import numpy as np
from TTTEnvironment import TTTEnvironment
from ourNN import NeuralNetwork
import copy
import os.path

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

    def __init__(self, a):
        # This is a dictionary. Information will be updated as playouts start. ['stateToString': 0]
        # 0 corresponds to position 0 on all the other arrays, 1 corresponds to position 1...hmmm.
        beginState = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.dictionary = {
            '0000000000000000000': 0  # empty board corresponds to position 0 on numpy arrays
        }
        #self.gameStateSeen = np.zeros(9) Commented because it seems obsolete

        self.childrenStateSeen = np.zeros((1, 9))  # This is a 2D array
        self.childrenStateWin = np.zeros((1, 9))  # This is a 2D array
        self.childrenNNEvaluation = np.zeros((1, 9))  # This is a 2D array
        self.childrenNNEvaluation[0] = a.predict(beginState)
        self.neuralNetwork = a

    def __init__(self, load=True):
        beginState = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.dictionary = {
            '0000000000000000000': 0  # empty board corresponds to position 0 on numpy arrays
        }
        # self.gameStateSeen = np.zeros(9) Commented because it seems obsolete

        self.childrenStateSeen = np.zeros((1, 9))  # This is a 2D array
        self.childrenStateWin = np.zeros((1, 9))  # This is a 2D array
        self.childrenNNEvaluation = np.zeros((1, 9))  # This is a 2D array
        self.neuralNetwork = NeuralNetwork(np.zeros((1, 19)), np.zeros((1, 9)), 50)

    def addToDictionary(self, position):
        self.dictionary[position]=len(self.dictionary)

    def simulation(self, originalPosition):
        # We store the information in the simulation in a temporary array, before adding everything to the database.
        turns = 0

        OstatesExplored = np.array([['0000000000000000000']])  # Store it as an array of directories for the seen stuff
        OactionsDone = np.zeros((1,9))
        XstatesExplored = np.array([['0000000000000000000']])  # Store it as an array of directories for the seen stuff
        XactionsDone = np.zeros((1,9))

        position = originalPosition

        #if game state is seen before,

        end=2 #0 if draw, 1 if X won, -1 if O won, 2 if keep going

        while(end==2):

            board = TTTEnvironment()
            board.state = TTTEnvironment.stringToState(board, position)
            TTTEnvironment.setValues(board)

            if position not in self.dictionary:
                self.initializePosition(position)

            nextPosition = list(position)

            """
            print("STATE: " + position)
            print(board.Xstate)
            print(board.Ostate)
            """

            #move = self.debugChooseMove(turns)
            move = self.chooseMove(position, board.legalMove(), 2**0.5)

            if ((int(board.turn)) % 2) == 0:  # If it's X to move
                nextPosition[move] = '1'
            if ((int(board.turn)) % 2) == 1:  # If it's O to move
                nextPosition[move + 9] = '1'
            nextPosition[-1] = str(1-int(nextPosition[-1]))

            nextPosition = ''.join(nextPosition)


            if ((int(board.turn)) % 2) == 0: # If it's X to move
                if turns != 0:
                    currentPosArray = np.array([[(position)]])
                    newAction=np.zeros((1,9))
                    newAction[0, move]=1
                    XstatesExplored=np.concatenate((XstatesExplored, currentPosArray), axis=0)
                    XactionsDone=np.concatenate((XactionsDone,newAction),axis=0)
                else:
                    XstatesExplored[0,0] = position
                    XactionsDone[0,move]=1
                #change tic tac toe board

            if ((int(board.turn)) % 2) == 1: # If it's O to move
                if turns != 0:
                    currentPosArray = np.array([[(position)]])
                    newAction = np.zeros((1, 9))
                    newAction[0, move] = 1
                    OstatesExplored = np.concatenate((OstatesExplored, currentPosArray), axis=0)
                    OactionsDone = np.concatenate((OactionsDone, newAction), axis=0)
                else:
                    OstatesExplored[0,0] = position
                    OactionsDone[0,move] = 1
                # change tic tac toe board
            turns += 1
            position = nextPosition

            board.state = TTTEnvironment.stringToState(board, position)
            TTTEnvironment.setValues(board)
            end=TTTEnvironment.check_Win(board)

        if end==1:
            for x in range(len(XstatesExplored)):
                dictionaryValue=self.dictionary[XstatesExplored[x,0]]
                self.childrenStateSeen[dictionaryValue]+=XactionsDone[x]
                self.childrenStateWin[dictionaryValue]+=XactionsDone[x]

            for x in range(len(OstatesExplored)):
                dictionaryValue=self.dictionary[OstatesExplored[x,0]]
                self.childrenStateSeen[dictionaryValue]+=OactionsDone[x]
                #no childrenStateWin because you lost

            #print('X wins')

        if end==-1:
            for x in range(len(XstatesExplored)):
                dictionaryValue=self.dictionary[XstatesExplored[x,0]]
                self.childrenStateSeen[dictionaryValue] += XactionsDone[x]
                # no childrenStateWin because you lost

            #print('O wins')

            for x in range(len(OstatesExplored)):
                dictionaryValue=self.dictionary[OstatesExplored[x,0]]
                self.childrenStateSeen[dictionaryValue] += OactionsDone[x]
                self.childrenStateWin[dictionaryValue] += OactionsDone[x]

        if end==0:
            for x in range(len(XstatesExplored)):
                dictionaryValue=self.dictionary[XstatesExplored[x,0]]
                self.childrenStateSeen[dictionaryValue]+=XactionsDone[x]
                self.childrenStateWin[dictionaryValue] += 0.5*XactionsDone[x]

            for x in range(len(OstatesExplored)):
                dictionaryValue=self.dictionary[OstatesExplored[x,0]]
                self.childrenStateSeen[dictionaryValue]+=OactionsDone[x]
                self.childrenStateWin[dictionaryValue] += 0.5*OactionsDone[x]

            #print('Draw')

        # just debugging
        #print('ends at', nextPosition)
        #print("X:")
        #print(board.Xstate)
        #print("O:")
        #print(board.Ostate)
        #if end==0:
            #print("TOTAL = ")
            #print(np.sum((board.Xstate, board.Ostate), axis=0))

    def runSimulations(self, sims, position):
        for i in range(sims):
            self.simulation(position)
        return self.dictionary

    #choose argmax per (P)UCT Algorithm
    def chooseMove(self, position, legalMoves, c, noise=False):
        index = self.dictionary[position]  # This returns a number based on the library

        # here we will assume that there is a legalMove function that works as followed:
        # If the first row of a tic tac toe row is all taken, then it returns [0,0,0,1,1,1,1,1,1]
        moveChoice = PUCT_Algorithm(self.childrenStateWin[index], self.childrenStateSeen[index], c * np.ones((1, 9)),
                                    np.sum(self.childrenStateSeen[index]), self.childrenNNEvaluation[index], legalMoves)


        #adding noise
        temp = TTTEnvironment()
        temp.state = TTTEnvironment.stringToState(temp, position)
        temp.setValues()
        noiseConstant = 0.1/(4*(1+np.sum(temp.Xstate.flatten())))
        if noise==True:
            noise = np.random.rand(1, 9)*noiseConstant*2-noiseConstant
            moveChoice = moveChoice + noise


        return np.argmax(moveChoice)

    def initializePosition(self, pos): #adds pos to dictionary and concatenates new layers to MCTS arrays
        self.addToDictionary(pos)
        self.childrenStateSeen = np.concatenate((self.childrenStateSeen,np.zeros((1, 9))), axis=0)
        self.childrenStateWin = np.concatenate((self.childrenStateWin, np.zeros((1, 9))), axis=0)
        board = TTTEnvironment()
        board.state = TTTEnvironment.stringToState(board, pos)
        board.setValues()
        winPercentages = self.neuralNetwork.predict(board.stateToArray())
        self.childrenNNEvaluation = np.concatenate((self.childrenNNEvaluation, winPercentages), axis=0)

    def updateEvals(self):
        beginState = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.dictionary = {
            '0000000000000000000': 0  # empty board corresponds to position 0 on numpy arrays
        }
        # self.gameStateSeen = np.zeros(9) Commented because it seems obsolete

        self.childrenStateSeen = np.zeros((1, 9))  # This is a 2D array
        self.childrenStateWin = np.zeros((1, 9))  # This is a 2D array
        self.childrenNNEvaluation = np.zeros((1, 9))  # This is a 2D array
        self.childrenNNEvaluation[0] = self.neuralNetwork.predict(beginState)


    def intToPos(self, integ):
        number = int(integ)
        number = str(number)
        length=len(number)
        for x in range(19 - length):
            number = '0' + number
        return number

    #returns the index of a move for competitive play
    def competitiveMove(self, sims, position):

        board = TTTEnvironment()
        board.state = TTTEnvironment.stringToState(board, position)
        board.setValues()

        self.runSimulations(sims, position)
        index = self.chooseMove(position, board.legalMove(), 0)
        return int(index)

    #returns the index of a move for training
    def trainingMove(self, sims, position):

        board = TTTEnvironment()
        board.state = TTTEnvironment.stringToState(board, position)
        board.setValues()

        self.runSimulations(sims, position)
        #index = np.argmax(self.childrenStateSeen[self.dictionary[position]])
        index = self.chooseMove(position, board.legalMove(), 2 ** 0.5, True)
        return int(index)

    def trainingGame(self, playouts, printLog=False):

        firstMove = True

        XstatesExplored = np.zeros((1, 19))  # Store it as an array of directories for the seen stuff
        XactionsDone = np.zeros((1, 9))
        XactionsWin = np.zeros((1, 9))
        OstatesExplored = np.zeros((1, 19))  # Store it as an array of directories for the seen stuff
        OactionsDone = np.zeros((1, 9))
        OactionsWin = np.zeros((1, 9))

        seriousGame = TTTEnvironment()
        while seriousGame.check_Win() == 2:
            # MAKES MOVE
            if printLog == True:
                index = self.competitiveMove(playouts, seriousGame.stateToString())
            else:
                index = self.trainingMove(playouts, seriousGame.stateToString())
            # push state first
            if int(seriousGame.turn) == 0:  # x moved
                if firstMove is True:
                    XstatesExplored = [seriousGame.stateToArray()]
                else:
                    newState = [seriousGame.stateToArray()]
                    XstatesExplored = np.concatenate((XstatesExplored, newState), axis=0)
            else:
                if firstMove is True:  # o moved
                    OstatesExplored = [seriousGame.stateToArray()]
                else:
                    newState = [seriousGame.stateToArray()]
                    OstatesExplored = np.concatenate((OstatesExplored, newState), axis=0)

            seriousGame.makeMove(index)  # make the move
            seriousGame.turn = str((int(seriousGame.turn) + 1) % 2)  # switch the turns.

            if int(seriousGame.turn) == 1:  # x moved
                newAction = np.zeros((1, 9))
                newAction[0, index] = 1
                if firstMove is True:
                    XactionsDone = newAction
                else:
                    XactionsDone = np.concatenate((XactionsDone, newAction), axis=0)

            else:  # o moved
                newAction = np.zeros((1, 9))
                newAction[0, index] = 1
                if firstMove is True:
                    OactionsDone = newAction
                    firstMove = False
                else:
                    OactionsDone = np.concatenate((OactionsDone, newAction), axis=0)

            seriousGame.updateState()
            if printLog == True:
                self.printBoard(seriousGame.Xstate, seriousGame.Ostate)
                print("-------")

        if printLog == True:
            print(seriousGame.check_Win())
            print(seriousGame.Xstate)
            print(seriousGame.Ostate)
        print(seriousGame.stateToString())
        if seriousGame.check_Win() == 0:
            XactionsWin = XactionsDone * 0.5
            OactionsWin = OactionsDone * 0.5
        if seriousGame.check_Win() == 1:
            XactionsWin = XactionsDone * 1
            OactionsWin = OactionsDone * 0
        if seriousGame.check_Win() == -1:
            XactionsWin = XactionsDone * 0
            OactionsWin = OactionsDone * 1

        # now, create the training data.
        inputs = np.concatenate((XstatesExplored, OstatesExplored), axis=0)
        outputSeen = np.concatenate((XactionsDone, OactionsDone), axis=0)
        outputWin = np.concatenate((XactionsWin, OactionsWin), axis=0)

        return inputs, outputSeen, outputWin

    def createDatabaseForNN(self, games, playouts):
        inputs = np.zeros((1, 19))
        outputSeen = np.zeros((1, 19))
        outputWin = np.zeros((1, 19))
        for i in range(games):
            newInputs, newOutputSeen, newOutputWin = self.trainingGame(playouts)
            if i == 0:
                inputs = newInputs
                outputSeen = newOutputSeen
                outputWin = newOutputWin
            else:
                #otherwise, check if input is already in dataset.
                removeDirectories = []
                for k in range(len(inputs)):
                    for j in range(len(newInputs)):
                        if np.sum(abs(inputs[k]-newInputs[j])) == 0:
                            #If information is already in dataset, edit existing data
                            outputWin[k] = outputWin[k] + newOutputWin[j]
                            outputSeen[k] = outputSeen[k] + newOutputSeen[j]
                            removeDirectories.append(j)
                removeDirectories.sort()
                while len(removeDirectories) > 0:
                    index = removeDirectories.pop()
                    newInputs = np.delete(newInputs, index, axis=0)
                    newOutputWin = np.delete(newOutputWin, index, axis=0)
                    newOutputSeen = np.delete(newOutputSeen, index, axis=0)
                inputs = np.concatenate((inputs, newInputs), axis=0)
                outputWin = np.concatenate((outputWin, newOutputWin), axis=0)
                outputSeen = np.concatenate((outputSeen, newOutputSeen), axis=0)

            print("Game " + str(i+1) + " processed.")

        outputs = np.divide(outputWin, outputSeen, out=np.zeros_like(outputWin), where=outputSeen != 0)
        return inputs, outputs

    def printBoard(self, a, b):
        for i in range(3):
            for j in range(3):
                if (a[i][j]==1):
                    print("X", ' ', end='')
                elif (b[i][j] == 1):
                     print("O", ' ', end='')
                else:
                    print("+", " ", end='')
            print(" ")

    def nnMove(position,nn):
        #pos=list(map(int, (position)))
        input=np.reshape(position,(1,19))
        print(nn)
        output=nn.predict(input)
        pos="".join(str(position[c]) for c in range(len(position)))
        tempBoard=TTTEnvironment()
        tempBoard.state=tempBoard.stringToState(pos)
        tempBoard.setValues()
        legalOutput=np.multiply(output,tempBoard.legalMove())
        return int(np.argmax(legalOutput, axis=1))

    def playEachOtherStart(x, y):
        end=2
        board=TTTEnvironment()
        board.state=TTTEnvironment.stringToState(board,'0000000000000000000')
        TTTEnvironment.setValues(board)

        posArray=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        while end==2:
            if(posArray[18]==0):
                move=MonteCarlo.nnMove(posArray,x)
                posArray[move+posArray[18]*9]=1
                posArray[18]=1
                board.state=TTTEnvironment.stringToState(board,"".join(str(posArray[c]) for c in range(len(posArray))))
                board.setValues()
                end=board.check_Win()
                if end!=2 :
                    print(board.state)
                    return end
            else:
                move = MonteCarlo.nnMove(posArray,y)
                posArray[move + posArray[18] * 9] = 1
                posArray[18] = 0
                board.state = board.stringToState("".join(str(posArray[c]) for c in range(len(posArray))))
                board.setValues()
                end = board.check_Win()
                if end != 2:
                    return end
        print(board.state)
        return end

    def playEachOther(a, b, startMove):
        end=2
        board=TTTEnvironment()
        board.state=TTTEnvironment.stringToState(board,'0000000000000000000')
        TTTEnvironment.setValues(board)

        posArray=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
        posArray[startMove]=1
        while end==2:
            if(posArray[18]==0):
                move=MonteCarlo.nnMove(posArray,a)
                posArray[move+posArray[18]*9]=1
                posArray[18]=1
                board.state=TTTEnvironment.stringToState(board,"".join(str(posArray[c]) for c in range(len(posArray))))
                board.setValues()
                end=board.check_Win()
                if end!=2 :
                    print(board.state)
                    return end
            else:
                move = MonteCarlo.nnMove(posArray,b)
                posArray[move + posArray[18] * 9] = 1
                posArray[18] = 0
                board.state = board.stringToState("".join(str(posArray[c]) for c in range(len(posArray))))
                board.setValues()
                end = board.check_Win()
                if end != 2:
                    return end
        print(board.state)
        return end

    def testEachOther(x, y, trials):
        xScore=0
        for m in range(trials):
            xScore += MonteCarlo.playEachOther(x,y,m%9)
            xScore -= MonteCarlo.playEachOther(y,x,m%9)
        xScore += 3*MonteCarlo.playEachOtherStart(x,y)
        xScore -= 3*MonteCarlo.playEachOtherStart(y,x)
        if xScore >= 0:
            print("New network was of equal strength or stronger, obtained a score of +", xScore)
            return x
        else:
            print("New network was weaker, obtained a score of +", xScore)
            return y

    def storeMCTSandNN(self):
        np.save("weight_1.npy", self.neuralNetwork.weight_1)
        np.save("weight_2.npy", self.neuralNetwork.weight_2)
        np.save("weight_3.npy", self.neuralNetwork.weight_3)
        np.save("weight_4.npy", self.neuralNetwork.weight_4)
        np.save("weight_5.npy", self.neuralNetwork.weight_5)
        np.save("bias_1.npy", self.neuralNetwork.bias_1)
        np.save("bias_2.npy", self.neuralNetwork.bias_2)
        np.save("bias_3.npy", self.neuralNetwork.bias_3)
        np.save("bias_4.npy", self.neuralNetwork.bias_4)
        np.save("bias_5.npy", self.neuralNetwork.bias_5)

        np.save("dictionary.npy", self.dictionary)
        np.save("childrenStateSeen.npy", self.childrenStateSeen)
        np.save("childrenStateWin.npy", self.childrenStateWin)
        np.save("childrenNNEvaluation.npy", self.childrenNNEvaluation)

    def loadMCTSandNN(self):
        loadBrain=NeuralNetwork(np.zeros((1,19)),np.zeros((1,9)), 50)
        if os.path.exists("bias_1.npy"):
            loadBrain.bias_1 = np.load("bias_1.npy")
        if os.path.exists("bias_2.npy"):
            loadBrain.bias_2 = np.load("bias_2.npy")
        if os.path.exists("bias_3.npy"):
            loadBrain.bias_3 = np.load("bias_3.npy")
        if os.path.exists("bias_4.npy"):
            loadBrain.bias_4 = np.load("bias_4.npy")
        if os.path.exists("bias_5.npy"):
            loadBrain.bias_5 = np.load("bias_5.npy")

        if os.path.exists("weight_1.npy"):
            loadBrain.weight_1 = np.load("weight_1.npy")
        if os.path.exists("weight_2.npy"):
            loadBrain.weight_2 = np.load("weight_2.npy")
        if os.path.exists("weight_3.npy"):
            loadBrain.weight_3 = np.load("weight_3.npy")
        if os.path.exists("weight_4.npy"):
            loadBrain.weight_4 = np.load("weight_4.npy")
        if os.path.exists("weight_5.npy"):
            loadBrain.weight_5 = np.load("weight_5.npy")

        self.dictionary = np.load("dictionary.npy").item()
        self.childrenStateSeen = np.load("childrenStateSeen.npy")
        self.childrenStateWin = np.load("childrenStateWin.npy")
        self.childrenNNEvaluation = np.load("childrenNNEvaluation.npy")

        self.neuralNetwork = loadBrain



    def printPredictions(self):
        temporaryBoard=TTTEnvironment()
        for pos in self.dictionary:
            temporaryBoard.state=temporaryBoard.stringToState(pos)
            temporaryBoard.setValues()
            print(pos,": ", self.chooseMove(pos,temporaryBoard.legalMove(),2**0.5,False))

#TESTING THE SELF-LEARNING PROCESS

brain = NeuralNetwork(np.zeros((1, 19)), np.zeros((1, 9)), 50)
#x = MonteCarlo(brain)
x = MonteCarlo(True)
x.loadMCTSandNN()
print("GAMES BY INITIAL NET")
x.trainingGame(5000, True)

for i in range(3000):
    print("GENERATION " + str(i+1))
    #450 games, 25 playouts for each move
    inputs, outputs = x.createDatabaseForNN(250, 80)
    previousBrain = copy.deepcopy(brain)
    brain = NeuralNetwork(inputs, outputs, 50)
    print(len(inputs))
    print("Testing new MCTS...")
    print("Training Network with previous data...")
    brain.trainNetwork(800, 0.008)
    print("Comparing New Neural Net...")
    brain = MonteCarlo.testEachOther(brain, previousBrain, 9)
    print("Self-learning is complete.")
    correct = (np.argmax(brain.predict(inputs), axis=1) == np.argmax(outputs, axis=1)).sum()
    print("Accuracy: ", correct/len(inputs))
    print("Total datasets: ", len(inputs))
    #Update the network onto MCTS
    #print("Printing Move Choices")
    #x.printPredictions()
    x.storeMCTSandNN()
    x.neuralNetwork = brain
    x.updateEvals()

    x.runSimulations(5000, '0000000000000000000')
    print("GAMES BY NEW NETWORK")
    x.trainingGame(5000, True)
