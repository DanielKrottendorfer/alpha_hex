
from hex_engine import hexPosition

import mcts

if(__name__ == "__main__"):

    myboard = hexPosition(size=4)
    myboard.printBoard()

    while myboard.winner == 0:
        
        ms = mcts.mctsagent(state = myboard)
        ms.search(1.0)
        move = ms.best_move()

        if move == -1 :
            break
        
        myboard.makeMoove(move,1)

        
        myboard.playRandom(2)
        
        myboard.printBoard()
    