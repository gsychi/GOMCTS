# w stands for # of wins, n stands for number of times node has been visited.
# N stands for number of times parent node is visited, and c is just an exploration constant that can be tuned.
# Q is the evaluation from 0 to 1 of the neural network

def UCT_Algorithm(w, n, c, N, q):
    # Provides a win rate score from 0 to 1
    selfPlayEvaluation = 0
    if n != 0:
        selfPlayEvaluation = w / n
    nnEvaluation = q
    winRate = (nnEvaluation + selfPlayEvaluation) / 2

    # Exploration
    exploration = c * math.sqrt(math.log(N) / n)

    UCT = winRate + exploration
    return UCT
