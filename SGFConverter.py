#This is a SGF Converter. It reads a sgf file and returns the list of moves that happened.
from GoEnvironment import GoEnvironment
from ourNN import NeuralNetwork
import numpy as np
from tempfile import TemporaryFile
import os.path

def findCoordinate(blah):
    sgfCoordinates = list("abcdefghi")
    for i in range(len(sgfCoordinates)):
        if blah == sgfCoordinates[i]:
            return i
    return -1

def returnAllMoves(string):

    actualCoordinates = list("ABCDEFGHJ")

    f = open(string).read()
    g = f.splitlines()
    h = []

    for _ in range(len(g)):
        if g[_][:2] == ";B" or g[_][:2] == ";W":
            if g[_][3:5] == "]":
                h.append("PASS")
            else:
                row = actualCoordinates[findCoordinate(g[_][3])]
                column = 9-findCoordinate(g[_][4])

                h.append(row+str(column))
    return h

def scrapeGame(game, result):
    go = GoEnvironment()

    #moves are downloaded
    a = returnAllMoves(game)
    print(a)

    statesExplored = np.zeros((1, 163))  # Store it as an array of directories for the seen stuff
    statesWin = np.zeros((1, 82))

    for i in range(len(a)):
        #turns start from 0...onwards.

        state = [go.boardToState()]

        #before move is made
        if go.turns%2 == 0:
            go.customMove(a[i], "B")
            #if result == 1:
            action = np.zeros((1, 82))
            action[0][go.directory(a[i])] = 1
            if go.turns == 0:
                statesExplored = state
                statesWin = action
            else:
                statesExplored = np.concatenate((statesExplored, state), axis=0)
                statesWin = np.concatenate((statesWin, action), axis=0)


        else:
            go.customMove(a[i], "W")
            #if result == -1:
            action = np.zeros((1, 82))
            action[0][go.directory(a[i])] = 1
            if go.turns == 1:
                statesExplored = state
                statesWin = action
            else:
                statesExplored = np.concatenate((statesExplored, state), axis=0)
                statesWin = np.concatenate((statesWin, action), axis=0)

        #update the board
        go.updateBoard()
        go.turns += 1

        #print the board.
        go.printBoard()

    return statesExplored, statesWin

def createDatabase(games, results, alreadyIn = np.zeros((1,163)), alreadyOut = np.zeros((1,82))):
    inputs = alreadyIn
    outputs = alreadyOut

    for i in range(len(games)):
        newInputs, newOutputs = scrapeGame(games[i],results[i])
        if i == 0:
            inputs = newInputs
            outputs = newOutputs
        else:
            # otherwise, check if input is already in dataset.
            removeDirectories = []
            for k in range(len(inputs)):
                for j in range(len(newInputs)):
                    if np.sum(abs(inputs[k] - newInputs[j])) == 0:
                        # If information is already in dataset, edit existing data
                        outputs[k] = outputs[k] + newOutputs[j]
                        removeDirectories.append(j)
            removeDirectories.sort()
            while len(removeDirectories) > 0:
                index = removeDirectories.pop()
                newInputs = np.delete(newInputs, index, axis=0)
                newOutputs = np.delete(newOutputs, index, axis=0)
            inputs = np.concatenate((inputs, newInputs), axis=0)
            outputs = np.concatenate((outputs, newOutputs), axis=0)

    #do something specific for the output.
    for i in range(len(outputs)):
        outputs[i] = outputs[i]/np.sum(outputs[i])

    return inputs, outputs


games = ["trainingGame1.sgf","trainingGame2.sgf","trainingGame3.sgf","trainingGame4.sgf","trainingGame5.sgf","trainingGame6.sgf", "trainingGame7.sgf", "trainingGame8.sgf", "trainingGame9.sgf", "trainingGame10.sgf", "trainingGame11.sgf"]
results = [-1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1]


training = False

inputs = np.zeros((1, 163))
outputs = np.zeros((1, 82))

if training:
    inputs, outputs = createDatabase(games, results)
    print(inputs.shape)

brain = NeuralNetwork(inputs, outputs, 50)
#load nn information so it can start from the same place for training
if os.path.exists("bias_1.npy"):
    brain.bias_1 = np.load("bias_1.npy")
if os.path.exists("bias_2.npy"):
    brain.bias_2 = np.load("bias_2.npy")
if os.path.exists("bias_3.npy"):
    brain.bias_3 = np.load("bias_3.npy")
if os.path.exists("bias_4.npy"):
    brain.bias_4 = np.load("bias_4.npy")
if os.path.exists("bias_5.npy"):
    brain.bias_5 = np.load("bias_5.npy")

if os.path.exists("weight_1.npy"):
    brain.weight_1 = np.load("weight_1.npy")
if os.path.exists("weight_2.npy"):
    brain.weight_2 = np.load("weight_2.npy")
if os.path.exists("weight_3.npy"):
    brain.weight_3 = np.load("weight_3.npy")
if os.path.exists("weight_4.npy"):
    brain.weight_4 = np.load("weight_4.npy")
if os.path.exists("weight_5.npy"):
    brain.weight_5 = np.load("weight_5.npy")


print("Start training...")
if training:
    for i in range(500):
        print("Epoch ", str(i+1))
        brain.trainNetwork(200, 0.005)
        correct = (np.argmax(brain.predict(inputs), axis=1) == np.argmax(outputs, axis=1)).sum()
        print("Accuracy: ", correct/len(inputs))
        print("Total datasets: ", len(inputs))
    print("--FINISH TRAINING--")

#save nn information
np.save("weight_1.npy", brain.weight_1)
np.save("weight_2.npy", brain.weight_2)
np.save("weight_3.npy", brain.weight_3)
np.save("weight_4.npy", brain.weight_4)
np.save("weight_5.npy", brain.weight_5)
np.save("bias_1.npy", brain.bias_1)
np.save("bias_2.npy", brain.bias_2)
np.save("bias_3.npy", brain.bias_3)
np.save("bias_4.npy", brain.bias_4)
np.save("bias_5.npy", brain.bias_5)

#We have now trained our neural network.

newBoard = GoEnvironment()

humanBlack = False
humanWhite = True

newBoard.printBoard()

for i in range(100):
    while newBoard.checkWin() == 2:
        predictedMove = np.argmax(brain.predict(newBoard.boardToState())*newBoard.legalMoves())
        #print(predictedMove)
        if newBoard.turns % 2 == 0:
            if not humanBlack:
                newBoard.customMoveNumber(predictedMove, "B")
            else:
                humanMove = input("Choose your coordinate:")
                newBoard.customMove(humanMove, "B")

        else:
            if not humanWhite:
                newBoard.customMoveNumber(predictedMove, "W")
            else:
                humanMove = input("Choose your coordinate:")
                newBoard.customMove(humanMove, "W")

        newBoard.updateBoard()
        newBoard.turns = newBoard.turns + 1
        newBoard.printBoard()

print(newBoard.printScore())
