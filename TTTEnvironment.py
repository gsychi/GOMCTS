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
