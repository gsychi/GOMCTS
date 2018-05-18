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
        matrix=x%9 #Tells whether to look at matrix X or matrix O
        x=int((x-x%9)/2)
        column=x%3 #Tells you the column of the array
        row=int((x-column)/3)%3 #Tells you the row of the array
        number=number+str(int(temp[matrix][row,column]))
    number=number+str(int(temp[2]))
    return int(number)

def numberToState(number):
    number=str(number)
    stateX=np.array(3,3)
    stateO=np.array(3,3)
    Turn=0
    gameState=stateX, stateO, turn
    for x in range(18):
        matrix=x%9 #Tells whether to look at matrix X or matrix O
        y=x-9*int(x/9)
        column=y%3 #Tells you the column of the array
        row=int((y-column)/3) #Tells you the row of the array
        gameState[y][row,column]=number[x]
    gameState[2]=number[-1]
