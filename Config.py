import argparse


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--epochs' , help = "Number of Epochs",type=int,default=1)
    parser.add_argument( '--cell_type', help="Choices:['RNN','GRU','LSTM']", type=str, default='GRU')
    parser.add_argument( '--batch_size', help="Batch size", type=int, default=128)
    parser.add_argument( '--optimizer', help = 'choices: [ "adam", "nadam"]', type=str, default = 'adam')
    parser.add_argument( '--learning_rate', help = 'Learning rate', type=float, default=0.0001)
    parser.add_argument( '--embedding_size', help='embedding SIZE', type=int, default=512)
    parser.add_argument( '--hidden_size', help='hidden layer size',type=int, default=512)
    parser.add_argument( '--dropout', help='choices:[0,0.2,0.3]',type=float, default=0.1)
    parser.add_argument( '--num_layers', help='Number of Layers',type=int, default=3)
    parser.add_argument( '--bidirectional', help='Choices:["True","False"]',type=str, default="False")
    parser.add_argument( '--teacher_forcing', help='choices:[0,0.2,0.3,0.5,0.7]',type=float, default=0.7)


    args = parser.parse_args()

    return args