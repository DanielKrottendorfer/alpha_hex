
from hex_engine import hexPosition

import mcts

if(__name__ == "__main__"):

    myboard = hexPosition(size=4)
    myboard.printBoard()

    while myboard.winner == 0:
        
        ms = mcts.mctsagent(state = myboard)
        ms.search(1.0)        
        myboard.makeMoove(ms.best_move(),1)
        
        myboard.playRandom()
        
        myboard.printBoard()
        
        myboard.calc_winner()
    