import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, input_size=26):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.embedding = nn.Embedding(self.input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size)
        # self.dropout = nn.Dropout(dropout_p)

    def forward_pass(self, input):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward_pass(self,input,hidden):
        decoder_embedding = self.embedding(input)
        activation = F.relu(decoder_embedding)
        output , hidden = self.rnn(activation,hidden)
        output = self.out(output)
        return output,hidden
    
    # TODO 
    # Lttile more to do``