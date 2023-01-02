import mcts
import hex_engine
import torch
size = 3

if(__name__ == "__main__"):
    print("hello")
    
    for i in range(0,100):

        myboard = hex_engine.hexPosition(size=size)
        myboard.makeMoove((1,1),1)

        running_loss = 0.0
        while myboard.winner == 0:

            ms = mcts.mctsagent(state = myboard)
            ms.search(2.0)        
            best_move = ms.best_move()            

            y = torch.nn.functional.normalize( torch.tensor(ms.get_float_matrix()))
            print(y)
            print()

        