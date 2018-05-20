import numpy as np

#X goes first

Xstate = np.zeros((3,3)) #0 if no X there, 1 if X there
Ostate = np.zeros((3,3)) #0 if no O there, 0 it O there
turn=0 #0 for X to move, 1 for O to move
state=Xstate, Ostate, turn

def stateToNumber(gameState):
    temp=gameState[0],gameState[1],gameState[2]
    number="" #19 places- 9 for X, 9 for O, 1 for turn
    for x in range(18):
        #Tells whether to look at matrix X or matrix O
        matrix=int((x-(x%9))/9)
        column=x%3 #Tells you the column of the array
        row=int((x-column)/3)%3 #Tells you the row of the array
        number=number+str(int(temp[matrix][row,column]))
    number=number+str(int(temp[2]))
    return int(number)

def numberToState(number):
    length=int(np.log10(number+1))+1
    number=str(number)
    for x in range(19-length):
        number='0'+number
    number=str(number)
    stateX=np.ndarray((3,3))
    stateO=np.ndarray((3,3))
    turn=0
    for x in range(18):
        matrix=x%9 #Tells whether to look at matrix X or matrix O
        matrix = int((x - (x % 9)) / 9)
        column = x % 3  # Tells you the column of the array
        row = int((x - column) / 3) % 3  # Tells you the row of the array
        if(matrix==0):
            stateX[row, column] = number[x]
        if(matrix==1):
            stateO[row, column] = number[x]
    turn=number[-1]
    gameState = stateX, stateO, turn
    return(gameState)

def makeMove(position):
    if turn==0:
        X_flat = Xstate.flatten
        X_flat[position] = 1
        Xstate = X_flat.reshape(3,3)

    else:
        Y_flat = Ystate.flatten
        Y_flat[position] = 1
        Ystate = Y_flat.reshape(3, 3)


def legalMove():
    Board_State = (Xstate+Ostate).flatten
    return 1-Board_State

def undoMove(position):
    if turn==0:
        X_flat = Xstate.flatten
        X_flat[position] = 0
        Xstate = X_flat.reshape(3,3)

    else:
        Y_flat = Ystate.flatten
        Y_flat[position] = 0
        Ystate = Y_flat.reshape(3, 3)

def check_Win():
    X_flat = Xstate.flatten
    for i in range(3):
        if X_flat[i*3] == 1 and X_flat[i*3 + 1] == 1 and X_flat[i*3 + 2] == 1:
            return 1
        elif X_flat[i] == 1 and X_flat[i + 3] == 1 and X_flat[i + 6] == 1:
            return 1

        if X_flat[0] == 1 and X_flat[4] == 1 and X_flat[8] == 1:
            return 1
        elif X_flat[2] == 1 and X_flat[4] == 1 and X_flat[6] == 1:
            return 1

    Y_flat = Ystate.flatten
    for i in range(3):
        if Y_flat[i*3] == 1 and Y_flat[i*3 + 1] == 1 and Y_flat[i*3 + 2] == 1:
            return -1
        elif Y_flat[i] == 1 and Y_flat[i + 3] == 1 and Y_flat[i + 6] == 1:
            return -1

        if Y_flat[0] == 1 and Y_flat[4] == 1 and Y_flat[8] == 1:
            return -1
        elif Y_flat[2] == 1 and Y_flat[4] == 1 and Y_flat[6] == 1:
            return -1

    if np.sum(legalMove()) != 0:
        return 2

    return 0

