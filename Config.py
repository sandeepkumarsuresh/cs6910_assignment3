"""
Parse command-line arguments using argparse.

"""


import argparse


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--epochs' , help = "Number of Epochs",type=int,default=10)
    parser.add_argument( '--cell_type', help="Choices:['RNN','GRU','LSTM']", type=str, default='LSTM')
    parser.add_argument( '--batch_size', help="Batch size", type=int, default=128)
    parser.add_argument( '--optimizer', help = 'choices: [ "adam", "nadam"]', type=str, default = 'nadam')
    parser.add_argument( '--learning_rate', help = 'Learning rate', type=float, default=0.001)
    parser.add_argument( '--embedding_size', help='embedding SIZE', type=int, default=512)
    parser.add_argument( '--hidden_size', help='hidden layer size',type=int, default=512)
    parser.add_argument( '--dropout', help='choices:[0,0.2,0.3]',type=float, default=0.1) # default 0.1 . Put 0 for attn
    parser.add_argument( '--num_layers', help='Number of Layers',type=int, default=4)
    parser.add_argument( '--bidirectional', help='Choices:["True","False"]',type=bool, default="True")
    parser.add_argument( '--teacher_forcing', help='choices:[0,0.2,0.3,0.5,0.7]',type=float, default=0.7)


    args = parser.parse_args()

    return args