import numpy as np

class GoEnvironment:

    def __init__(self):
        self.board = np.zeros((9,9,2))
        self.plies_before_board = np.zeros((3,9,9,2))
        self.turns = 0
        self.alphabet = 'ABCDEFGHJ'
        self.wChains = []
        self.wSurChains = []
        self.bChains = []
        self.bSurChains = []
        self.illegalKo = 999
        self.turnOfIllegalKo = 999

        self.gamelog = ';FF[4]\nGM[1]\nDT[2018-01-10]\nPB[EKALGO]\nPW[EKALGO]\nBR[30k]\nWR[30k]\nRE[]\nSZ[19]\nKM[7.5]\nRU[chinese]'

    def addToGameLog(self, i, j, black):
        alphabet = "abcdefghi"
        if black:
            self.gameLog += '\n;B[' + alphabet.split("")[i] + alphabet.split("")[j] + ']'
        else:
            self.gameLog+='\n;W['+alphabet.split("")[i]+alphabet.split("")[j]+']'

    def output(self, a):
        if a[0] == 1:
            return '●'
        elif a[1] == 1:
            return 'o'
        return '+'

    def flipAction180(self, move):
        directory = self.alphabet.index(move[:1])
        return self.alphabet.split("")[8-directory]+(10-int(move[1:]))

    def flipAction90CC(self, move):
        column = self.alphabet.index(move[:1])-5
        row = int(move[1:])-5
        newCol = -row+5
        newRow = column+5
        return self.alphabet.split("")[newCol-1]+(newRow+1)

    def flipAction90C(self,move):
        column = self.alphabet.index(move[:1]) - 5
        row = int(move[1:]) - 5
        newCol = row + 5
        newRow = -column + 5
        return self.alphabet.split("")[newCol - 1] + (newRow - 1)

    def mirrorY(self, move):
        directory = self.alphabet.index(move[:1])
        return self.alphabet.split("")[8-directory]+move[1:]

    def mirrorX(self, move):
        directory = self.alphabet.index(move[:1])
        return self.alphabet.split("")[directory] + (20 - int(move[1:]))

    def printBoard(self):
        print('CURRENT STATE:\n\n   A B C D E F G H J K L M N O P Q R S T')
        for i in range(9):
            if i>5:
                print(19-i, ' ')
            else:
                print(19-i, ' ')
            for j in range(9):
                print(self.output(self.board[i][j]),' ')
            print('\n')

    def customMove(self, move, colour):
        print("nothing")

    def makeMove(self, move):
        print("nothing")

    def OneBoardToState(self, board):
        output = []
        for i in range(9):
            for j in range(9):
                output[27*i+3*j] = board[i][j][0]
                output[27*i+3*j+1] = board[i][j][1]
                if output[27*i+3*j]==0 and output[27*i+3*j+1]==0:
                    output[27*i+3*j+2] = 1
        output[81*3]=self.turns%2
        #return np.concatenate(output,ConvolutionsNN.processed(board))

    def BoardToState(self):
        state = self.OneBoardToState(self.board)
        for i in range(len(self.plies_before_board)):
            state = np.concatenate(state, self.OneBoardToState(self.plies_before_board[i]))

        return state

    def surroundReq(self,i,j):
        ba = []
        if i!=0:
            ba.append(((i-1)*9+j))
        if j!=0:
            ba.append((i) * 9 + j - 1)
        if j!=8:
            ba.append((i) * 9 + j + 1)
        if i!=8:
            ba.append((i + 1) * 9 + j)

        return ba

    def searchSurroundings(self, i, j, k):
        required = 4
        stoneSurround = 0
        if i==0:
            required-=1
        elif self.board[i-1][j][k]==1 and self.board[i-1][j][k-1]==0:
            stoneSurround+=1
        if j==0:
            required-=1
        elif self.board[i][j-1][k]==1 and self.board[i][j-1][1-k]==0:
            stoneSurround+=1
        if i==8:
            required-=1
        elif self.board[i+1][j][k]==1 and self.board[i+1][j][1-k]==0:
            stoneSurround+=1
        if j==8:
            required-=1
        elif self.board[i][j+1][k]==1 and self.board[i][j+1][1-k]==0:
            stoneSurround+=1
        if stoneSurround == required:
            self.illegalKo = (i*9)+j
            self.turnOfIllegalKo = self.turns
            self.board[i][j][1 - k] = 0
            self.board[i][j][k] = 0
            print('Captured stone ' , ((i * 9) + j))

    def locateChains(self):
        print("tbc")
    def followChain(self, board, i, j, c):
        print("tbc")
    def findChains(self, board):
        print("tbc")
    def surArray(self, chains):
        output = np.ones(len(chains),1)
        for i in range(len(chains)):
            output[i] = self.surroundings(chains[i])
        return output

    # def surroundings(self, chain):
    #     surroundings = self.surroundReq(chain[0]/9,chain[0]%9)
    #     for i in range(len(chain)-1):
    #         surroundings =

    def ifSurroundedByW(self, require):
        covered = 0
        for i in range(len(require)):
            if self.board[require[i]/9][require[i]%9][1]==1:
                covered+=1
        return covered == len(require)

    def ifSurroundedByB(self, require):
        covered = 0
        for i in range(len(require)):
            if self.board[require[i]/9][require[i]%9][0]==1:
                covered+=1
        return covered == len(require)

    def removeStones(self, stones):
        for i in range(len(stones)):
            self.board[stones[i] / 9][stones[i] % 9][0] = 0;
            self.board[stones[i] / 9][stones[i] % 9][1] = 0;

    def captureDeadStonesW(self):
        for i in range(len(self.wChains)):
            if self.ifSurroundedByB(self.wSurChains[i]):
                self.removeStones(self.wChains[i])
                print('Captured chain ' , self.wChains[i])

        # for (int i = 0;i < wStones.size();i++) {
        # searchSurroundings(wStones.get(i)[0] / 19, wStones.get(i)[0] % 19, 0);
        # }

    def captureDeadStonesB(self):
        for i in range(len(self.bChains)):
            if self.ifSurroundedByW(self.bSurChains[i]):
                self.removeStones(self.bChains[i])
                print('Captured chain ' , self.bChains[i])

        # for (int i = 0;i < bStones.size();i++) {
        # searchSurroundings(bStones.get(i)[0] / 19, bStones.get(i)[0] % 19, 1);
        #
        # }

    def printAllChains(self):
        print('\n----\nFOR BLACK:\n---- ')
        for i in range(len(self.bChains)):
            print(self.bChains[i])
            print('Surrounding Stones: ' , self.bSurChains[i])
        print('\n-----\nFOR WHITE:\n---- ')
        for i in range(len(self.wChains)):
            print(self.wChains[i])
            print('Surrounding Stones: ' , self.wSurChains[i])



