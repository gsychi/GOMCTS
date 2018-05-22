import numpy as np


class TTTEnvironment:

    # X goes first
    def __init__(self):
        self.Xstate = np.zeros((3, 3))  # 0 if no X there, 1 if X there
        self.Ostate = np.zeros((3, 3))  # 0 if no O there, 0 it O there
        self.turn = 0  # 0 for X to move, 1 for O to move
        self.state = self.Xstate, self.Ostate, self.turn

    def setValues(self):
        self.Xstate, self.Ostate, self.turn=self.state

    def stateToNumber(gameState):
        temp = gameState[0], gameState[1], gameState[2]
        number = ""  # 19 places- 9 for X, 9 for O, 1 for turn
        for x in range(18):
            # Tells whether to look at matrix X or matrix O
            matrix = int((x - (x % 9)) / 9)
            column = x % 3  # Tells you the column of the array
            row = int((x - column) / 3) % 3  # Tells you the row of the array
            number = number + str(int(temp[matrix][row, column]))
        number = number + str(int(temp[2]))
        return int(number)

    def stateToString(gameState):
        temp = gameState[0], gameState[1], gameState[2]
        number = ""  # 19 places- 9 for X, 9 for O, 1 for turn
        for x in range(18):
            # Tells whether to look at matrix X or matrix O
            matrix = int((x - (x % 9)) / 9)
            column = x % 3  # Tells you the column of the array
            row = int((x - column) / 3) % 3  # Tells you the row of the array
            number = number + str(int(temp[matrix][row, column]))
        number = number + str(int(temp[2]))
        return number

    def numberToState(self, number):
        length = int(np.log10(number + 1)) + 1
        number = str(number)
        for x in range(19 - length):
            number = '0' + number
        number = str(number)
        stateX = np.ndarray((3, 3))
        stateO = np.ndarray((3, 3))
        turn = 0
        for x in range(18):
            matrix = x % 9  # Tells whether to look at matrix X or matrix O
            matrix = int((x - (x % 9)) / 9)
            column = x % 3  # Tells you the column of the array
            row = int((x - column) / 3) % 3  # Tells you the row of the array
            if (matrix == 0):
                stateX[row, column] = number[x]
            if (matrix == 1):
                stateO[row, column] = number[x]
        turn = number[-1]
        gameState = stateX, stateO, turn
        return (gameState)

    def stringToState(self, string):
        length = len(string)
        number=string[:]
        for x in range(19 - length):
            number = '0' + number
        number = str(number)
        stateX = np.ndarray((3, 3))
        stateO = np.ndarray((3, 3))
        turn = 0
        for x in range(18):
            matrix = x % 9  # Tells whether to look at matrix X or matrix O
            matrix = int((x - (x % 9)) / 9)
            column = x % 3  # Tells you the column of the array
            row = int((x - column) / 3) % 3  # Tells you the row of the array
            if (matrix == 0):
                stateX[row, column] = number[x]
            if (matrix == 1):
                stateO[row, column] = number[x]
        turn = number[-1]
        gameState = stateX, stateO, turn
        return (gameState)

    def makeMove(self,i):
        if self.turn == 0:
            X_flat = self.Xstate.flatten()
            X_flat[i] = 1
            self.Xstate = X_flat.reshape(3, 3)

        else:
            O_flat = self.Ostate.flatten()
            O_flat[i] = 1
            self.Ostate = O_flat.reshape(3, 3)

    def legalMove(self):
        Board_State = (self.Xstate + self.Ostate).flatten()
        return 1 - Board_State

    def undoMove(self,i):
        if self.turn == 0:
            X_flat = self.Xstate.flatten()
            X_flat[i] = 0
            self.Xstate = X_flat.reshape(3, 3)

        else:
            O_flat = self.Ostate.flatten()
            O_flat[i] = 0
            self.Ostate = O_flat.reshape(3, 3)

    def check_Win(self):
        X_flat = self.Xstate.flatten()
        for i in range(3):
            if X_flat[i * 3] == 1 and X_flat[i * 3 + 1] == 1 and X_flat[i * 3 + 2] == 1:
                return 1
            elif X_flat[i] == 1 and X_flat[i + 3] == 1 and X_flat[i + 6] == 1:
                return 1

            if X_flat[0] == 1 and X_flat[4] == 1 and X_flat[8] == 1:
                return 1
            elif X_flat[2] == 1 and X_flat[4] == 1 and X_flat[6] == 1:
                return 1

        O_flat = self.Ostate.flatten()
        for i in range(3):
            if O_flat[i * 3] == 1 and O_flat[i * 3 + 1] == 1 and O_flat[i * 3 + 2] == 1:
                return -1
            elif O_flat[i] == 1 and O_flat[i + 3] == 1 and O_flat[i + 6] == 1:
                return -1

            if O_flat[0] == 1 and O_flat[4] == 1 and O_flat[8] == 1:
                return -1
            elif O_flat[2] == 1 and O_flat[4] == 1 and O_flat[6] == 1:
                return -1

        if np.sum(self.legalMove()) != 0:
            return 2

        return 0

def main():
    newenv = TTTEnvironment()
    print(newenv.legalMove())
    newenv.makeMove(0)
    print(newenv.Xstate)
    newenv.turn = 1
    newenv.makeMove(3)
    print(newenv.Xstate)
    print(newenv.Ostate)
    print(newenv.check_Win())
    newenv.turn = 0
    newenv.makeMove(1)
    newenv.makeMove(2)
    print(newenv.check_Win())
