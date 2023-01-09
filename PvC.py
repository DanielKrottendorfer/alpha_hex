
import torch
from torch import nn
import torch.nn.functional as F
import hex_engine
import numpy as np
import mcts


import train




def translator (string):
    #This function translates human terminal input into the proper array indices.
    number_translated = 27
    letter_translated = 27
    names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(string) > 0:
        letter = string[0]
    if len(string) > 1:
        number1 = string[1]
    if len(string) > 2:
        number2 = string[2]
    for i in range(26):
        if names[i] == letter:
            letter_translated = i
            break
    if len(string) > 2:
        for i in range(10,27):
            if number1 + number2 == "{}".format(i):
                number_translated = i-1
    else:
        for i in range(1,10):
            if number1 == "{}".format(i):
                number_translated = i-1
    return (number_translated, letter_translated)
import my_model
model =  my_model.OthelloNNet(train.size)
if __name__ == '__main__':
    
    model_path = "./model.pt"
    file = torch.load(model_path)
    model.load_state_dict(file)
    model.eval()

    board = hex_engine.hexPosition(size=train.size)

    machine = False

    while True:
        
        if machine:
            b = np.array(board.get_float_state(),dtype=np.single)
            pred,v = model.forward(torch.tensor(b))
            actions = board.getActionSpace()
            
            max_i = actions[0]
            for j in range(1,len(actions)):
                if pred[max_i[0]][max_i[1]] <= pred[actions[j][0]][actions[j][1]]:
                    max_i = actions[j]
            
            if board.play(max_i):
                break
        else:
            
            ms = mcts.mctsagent(state = board)
            ms.search(roll_outs=900)        
            best_move = ms.best_move()            

            if board.play(best_move):
                break


        board.printBoard()
        

        human_input = translator(input("Enter your moove (e.g. 'A1'): "))

        if board.play(human_input):
            break
    


    board.printBoard()