#This is a SGF Converter. It reads a sgf file and returns the list of moves that happened.
from GoEnvironment import GoEnvironment
import numpy as np

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
    aResult = result #white won this game.
    print(a)

    statesExplored = np.zeros((1, 163))  # Store it as an array of directories for the seen stuff
    statesWin = np.zeros((1, 82))

    for i in range(len(a)):
        #turns start from 0...onwards.

        state = [go.boardToState()]

        #before move is made
        if go.turns%2 == 0:
            go.customMove(a[i], "B")
            if result == 1:
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
            if result == -1:
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

def createDatabaes(games, results):
    inputs = np.zeros((1, 163))
    outputs = np.zeros((1, 82))
    
    return "wow"
inputs, outputs = scrapeGame("trainingGame1.sgf",-1)
print(inputs.shape)
print(outputs.shape)