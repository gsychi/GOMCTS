import numpy as np

class GoEnvironment():

    def __init__(self):
        self.board = np.zeros((9,9,2))
        self.plies_before_board = np.zeros((3,9,9,2))
        self.turns = 0
        self.alphabet = 'ABCDEFGHJ'
        self.wStones = []
        self.wSurStones = []
        self.bStones = []
        self.bSurStones = []
        self.wChains = []
        self.wSurChains = []
        self.bChains = []
        self.bSurChains = []
        self.blankStones = []
        self.blankSurStones = []
        self.blankChains = []
        self.blankSurChains = []
        self.illegalKo = 999
        self.passesInARow = 0

        self.gamelog = ';FF[4]\nGM[1]\nDT[2018-01-10]\nPB[EKALGO]\nPW[EKALGO]\nBR[30k]\nWR[30k]\nRE[]\nSZ[9]\nKM[7.5]\nRU[chinese]'

    def boardToState(self):
        player = self.turns % 2
        return np.append(self.board.flatten(), player)

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
            return 'O'
        return '+'

    def flipAction180(self, move):
        directory = self.alphabet.index(move[:1])
        return list(self.alphabet)[8-directory]+(10-int(move[1:]))

    def flipAction90CC(self, move):
        column = self.alphabet.index(move[:1])-5
        row = int(move[1:])-5
        newCol = -row+5
        newRow = column+5
        return list(self.alphabet)[newCol-1]+(newRow+1)

    def flipAction90C(self,move):
        column = self.alphabet.index(move[:1]) - 5
        row = int(move[1:]) - 5
        newCol = row + 5
        newRow = -column + 5
        return list(self.alphabet)[newCol - 1] + (newRow - 1)

    def mirrorY(self, move):
        directory = self.alphabet.index(move[:1])
        return list(self.alphabet)[8-directory]+move[1:]

    def mirrorX(self, move):
        directory = self.alphabet.index(move[:1])
        return list(self.alphabet)[directory] + (20 - int(move[1:]))

    def printBoard(self):
        print('CURRENT STATE:\n\n   A  B  C  D  E  F  G  H  J')
        for i in range(9):
            if i>5:
                print(9-i, ' ', end='')
            else:
                print(9-i, ' ', end='')
            for j in range(9):
                print(self.output(self.board[i][j]),' ', end='')
            print('\n')

    def customMoveNumber(self, number, colour):
        if number != 81:
            if colour == 'B':
                turn = 0
            elif colour == 'W':
                turn = 1
            self.board[int(number/9)][int(number % 9)][turn] = 1
        else:
            self.passesInARow = self.passesInARow + 2

    def customMove(self, move, colour):
        if move != "PASS":
            directory = np.zeros(2)
            directory[0] = int(self.alphabet.index(move[:1]))
            directory[1] = 9-int(move[1:])
            if colour == 'B':
                turn = 0
            elif colour == 'W':
                turn = 1
            self.board[int(directory[1])][int(directory[0])][turn] = 1
        else:
            self.passesInARow = self.passesInARow + 2

    def directory(self, move):
        if move != "PASS":
            directory = np.zeros(2)
            directory[0] = int(self.alphabet.index(move[:1]))
            directory[1] = 9 - int(move[1:])
            return int(directory[0]*9 + directory[1])
        else:
            return 81

    def updateBoard(self):
        if self.passesInARow > 0:
            self.passesInARow = self.passesInARow - 1
        self.illegalKo = 999
        self.wStones.clear()
        self.wSurStones.clear()
        self.bStones.clear()
        self.bSurStones.clear()
        self.locateChains()
        if self.turns%2==1:
            self.captureDeadStonesB()
        else:
            self.captureDeadStonesW()

    def surroundReq(self,i,j): #return liberties of each stone
        ba = []
        if i!=0:
            ba.append(((i-1)*9+j))
        if j!=0:
            ba.append((i)*9+j-1)
        if j!=8:
            ba.append((i)*9+j+1)
        if i!=8:
            ba.append((i+1)*9+j)

        return ba

    def isIndividual(self,i,j,color, map): #return liberties of each stone
        flag = True
        if color == 0: #if finding for black
            if i!=0:
                if(map[i-1][j][0] == 1):
                    flag = False
            if j!=0:
                if (map[i][j-1][0] == 1):
                    flag = False
            if j!=8:
                if (map[i][j+1][0] == 1):
                    flag = False
            if i!=8:
                if (map[i + 1][j][0] == 1):
                    flag = False
        elif color == 1:
            if i!=0:
                if(map[i-1][j][1] == 1):
                    flag = False
            if j!=0:
                if (map[i][j-1][1] == 1):
                    flag = False
            if j!=8:
                if (map[i][j+1][1] == 1):
                    flag = False
            if i!=8:
                if (map[i + 1][j][1] == 1):
                    flag = False
        elif color == -1:
            if i!=0:
                if(map[i-1][j] == 1):
                    flag = False
            if j!=0:
                if (map[i][j-1] == 1):
                    flag = False
            if j!=8:
                if (map[i][j+1] == 1):
                    flag = False
            if i!=8:
                if (map[i + 1][j] == 1):
                    flag = False
        return flag

    def searchSurroundings(self, i, j, k):
        required = 4
        stoneSurround = 0
        i=int(i)
        j=int(j)
        k=int(k)
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
            self.board[i][j][1 - k] = 0
            self.board[i][j][k] = 0
            print('Captured stone ' , ((i * 9) + j))

    def locateChains(self):
        for i in range(9):
            for j in range(9):
                array = np.ones(9*i+j)
                if self.board[i][j][1]==1:
                    self.wStones.append(array)
                    self.wSurStones.append(self.surroundReq(i,j))
                if self.board[i][j][0]==1:
                    self.bStones.append(array)
                    self.bSurStones.append(self.surroundReq(i,j))

        white_stone_map = np.ones((9,9))
        black_stone_map = np.ones((9,9))
        for i in range(9):
            for j in range(9):
                white_stone_map[i][j] = int(self.board[i][j][1])
                black_stone_map[i][j] = int(self.board[i][j][0])

        self.wChains = self.findChains(white_stone_map)
        for i in range(9):
            for j in range(9):
                if i== 1 and j == 0:
                    print(self.isIndividual(i,j,1,self.board))
                if self.isIndividual(i,j,1,self.board) == True and self.board[i][j][1] == 1:
                    self.wChains.append([9*i+j])
        self.bChains = self.findChains(black_stone_map)
        for i in range(9):
            for j in range(9):
                if self.isIndividual(i,j,0,self.board) == True and self.board[i][j][0] == 1:
                    self.bChains.append([9*i+j])
        self.wSurChains = self.surArray(self.wChains)
        self.bSurChains = self.surArray(self.bChains)


    def followChain(self, board, i, j, c=[]):
        chain = list(c)  # This just forces it to pass by value. ignore when porting to javab
        if i >= len(board) or i < 0 or j >= len(board[0]) or j < 0:
            return chain
        if board[i][j] == 0 or i * 9 + j in chain:
            return chain
        chain.append(i * 9 + j)
        chain1 = self.followChain(board, i + 1, j, chain)
        chain2 = self.followChain(board, i, j + 1, chain1)
        chain3 = self.followChain(board, i - 1, j, chain2)
        chain4 = self.followChain(board, i, j - 1, chain3)
        return chain4

    def findChains(self, board):
        chains = []
        for i in range(9):
            for j in range(9):
                elem = board[i][j]
                if elem == 1:
                    if (j != 0 and board[i][j - 1] == 1) or (i != 0 and board[i - 1][j] == 1):
                        continue
                    chain = self.followChain(board, i, j)
                    if len(chain) > 1:
                        chains.append(chain)
        return chains

    def surArray(self, chains):
        output = []
        for i in range(len(chains)):
            output.append(self.surroundings(chains[i]))
        return output


    def surroundings(self, chain):
        wrongsurroundings = []
        for i in range(len(chain)):
            wrongsurroundings.append((self.surroundReq(int(int(chain[i]/9)),int(int(chain[i]%9)))))
        flat_sur = []

        for x in wrongsurroundings:
            for y in x:
                flat_sur.append(int(y))

        unique_list = []

        for i in flat_sur:
            if i not in unique_list:
                unique_list.append(i)


        sur = []

        for i in range(len(unique_list)):
            if np.in1d([unique_list[i]],chain) == False:
                sur.append(unique_list[i])


        return sur

    def ifSurroundedByW(self, require):
        covered = 0
        for i in range(len(require)):
            if self.board[int(require[i]/9)][int(require[i]%9)][1]==1:
                covered+=1
        return covered == len(require)

    def ifSurroundedByB(self, require):
        covered = 0
        #print('require', require)
        #print('ifsurroundedbyb',require)
        for i in range(len(require)):
            if self.board[int(require[i]/9)][int(require[i]%9)][0]==1:
                covered+=1
        #print('covered',covered)
        return covered == len(require)

    def removeStones(self, stones):
        print('removed',stones)
        #if the stone is an individual, then enforce illegalKo move.
        if len(stones)==1:
            self.illegalKo = stones[0]

        for i in range(len(stones)):
            self.board[int(stones[i] / 9)][int(stones[i] % 9)][0] = 0
            self.board[int(stones[i] / 9)][int(stones[i] % 9)][1] = 0

    def captureDeadStonesW(self):
        #print('wchain', self.wChains)
        #print('wchain length', len(self.wChains))
        #print('wsurchain',self.wSurChains)
        for i in range(len(self.wChains)):
            if self.ifSurroundedByB(self.wSurChains[i]):
                self.removeStones(self.wChains[i])
                #print('Captured chain ' , self.wChains[i])

        #for i in range(len(self.wStones)):
            #self.searchSurroundings(int(self.wStones[i][0]/9),int(self.wStones[i][0]%9), 0)

    def captureDeadStonesB(self):
        #print('bchain',self.bChains)
        for i in range(len(self.bChains)):
            if self.ifSurroundedByW(self.bSurChains[i]):
                self.removeStones(self.bChains[i])
                #print('Captured chain ' , self.bChains[i])

        #for i in range(len(self.bStones)):
            #self.searchSurroundings(int(self.bStones[i][0]/9),int(self.bStones[i][0]%9), 1)

    def inaccStones(self, stones, wrongSurround):
        ky = []
        for i in range(len(wrongSurround)):
            if np.in1d(stones, wrongSurround[i]) == False:
                ky.append(wrongSurround[i])
        return ky

    def printAllChains(self):
        print('\n----\nFOR BLACK:\n---- ')
        for i in range(len(self.bChains)):
            print(self.bChains[i])
            print('Surrounding Stones: ' , self.bSurChains[i])
        print('\n-----\nFOR WHITE:\n---- ')
        for i in range(len(self.wChains)):
            print(self.wChains[i])
            print('Surrounding Stones: ' , self.wSurChains[i])

    def printScore(self):
        score_white = 5.5
        score_black = 0
        for i in range(9):
            for j in range(9):
                score_white += int(self.board[i][j][1])
                score_black += int(self.board[i][j][0])

        blank_stone_map = np.ones((9,9))

        for i in range(9):
            for j in range(9):
                if int(self.board[i][j][1]) == 0 and int(self.board[i][j][0]) == 0:
                    blank_stone_map[i][j] = 1
                else:
                    blank_stone_map[i][j] = 0

        #print(blank_stone_map)
        self.blankChains = self.findChains(blank_stone_map)
        for i in range(9):
            for j in range(9):
                if self.isIndividual(i,j,-1,blank_stone_map) == True and blank_stone_map[i][j] == 1:
                    self.blankChains.append([9*i+j])

        unique = []
        for i in self.blankChains:
            i.sort(key=int)
            if i not in unique:
                unique.append(i)

        self.blankChains = unique
        #print(self.blankChains)
        self.blankSurChains = self.surArray(self.blankChains)

        for i in range(len(self.blankSurChains)):
            if self.ifSurroundedByB(self.blankSurChains[i]) == True:
                score_black+=len(self.blankChains[i])
            elif self.ifSurroundedByW(self.blankSurChains[i]) == True:
                score_white+=len(self.blankChains[i])

        print('Black Score: ' , score_black)
        print('White Score: ' , score_white)
        if score_black > score_white:
            print("Congrats to Black for winning!")
        else:
            print("Congrats to White for winning!")

    def checkWin(self):
        score_white = 5.5
        score_black = 0
        for i in range(9):
            for j in range(9):
                score_white += int(self.board[i][j][1])
                score_black += int(self.board[i][j][0])

        blank_stone_map = np.ones((9,9))

        for i in range(9):
            for j in range(9):
                if int(self.board[i][j][1]) == 0 and int(self.board[i][j][0]) == 0:
                    blank_stone_map[i][j] = 1
                else:
                    blank_stone_map[i][j] = 0

        #print(blank_stone_map)
        self.blankChains = self.findChains(blank_stone_map)
        for i in range(9):
            for j in range(9):
                if self.isIndividual(i,j,-1,blank_stone_map) == True and blank_stone_map[i][j] == 1:
                    self.blankChains.append([9*i+j])

        unique = []
        for i in self.blankChains:
            i.sort(key=int)
            if i not in unique:
                unique.append(i)

        self.blankChains = unique
        #print(self.blankChains)
        self.blankSurChains = self.surArray(self.blankChains)

        for i in range(len(self.blankSurChains)):
            if self.ifSurroundedByB(self.blankSurChains[i]) == True:
                score_black+=len(self.blankChains[i])
            elif self.ifSurroundedByW(self.blankSurChains[i]) == True:
                score_white+=len(self.blankChains[i])

        #print('Black Score: ' , score_black)
        #print('White Score: ' , score_white)

        if self.passesInARow == 2:
            if score_black>score_white:
                return 1
            else:
                return -1
        else:
            return 2

    def legalMoves(self):
        filledUpBoard = np.sum(self.board, axis=2).flatten()
        legalMoves = 1 - filledUpBoard
        if self.illegalKo != 999:
            legalMoves[self.illegalKo] = 0
        passMove = 1
        return [np.append(legalMoves,passMove)]
