#This is a SGF Converter. It reads a sgf file and returns the list of moves that happened.
from GoEnvironment import GoEnvironment
from ourNN import NeuralNetwork
import numpy as np
from tempfile import TemporaryFile

weights1 = TemporaryFile()
weights2 = TemporaryFile()
weights3 = TemporaryFile()
weights4 = TemporaryFile()
weights5 = TemporaryFile()
bias1 = TemporaryFile()
bias2 = TemporaryFile()
bias3 = TemporaryFile()
bias4 = TemporaryFile()
bias5 = TemporaryFile()

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

inputs, outputs = createDatabase(games, results)
print(inputs.shape)

#print(outputs.shape)
#print(np.sum(outputs, axis=1))


brain = NeuralNetwork(inputs, outputs, 50)
#load nn information so it can start from the same place for training
brain.weight_1 = np.load(weights1)
brain.weight_2 = np.load(weights2)
brain.weight_3 = np.load(weights3)
brain.weight_4 = np.load(weights4)
brain.weight_5 = np.load(weights5)
brain.bias_1 = np.load(bias1)
brain.bias_2 = np.load(bias2)
brain.bias_3 = np.load(bias3)
brain.bias_4 = np.load(bias4)
brain.bias_5 = np.load(bias5)

print("Start training...")
for i in range(200):
    print("Epoch ", str(i+1))
    brain.trainNetwork(200, 0.005)
    correct = (np.argmax(brain.predict(inputs), axis=1) == np.argmax(outputs, axis=1)).sum()
    print("Accuracy: ", correct/len(inputs))
    print("Total datasets: ", len(inputs))
print("--FINISH TRAINING--")

#save nn information
np.save(brain.weight_1, weights1)
np.save(brain.weight_2, weights2)
np.save(brain.weight_3, weights3)
np.save(brain.weight_4, weights4)
np.save(brain.weight_5, weights5)
np.save(brain.bias_1, bias1)
np.save(brain.bias_2, bias2)
np.save(brain.bias_3, bias3)
np.save(brain.bias_4, bias4)
np.save(brain.bias_5, bias5)
