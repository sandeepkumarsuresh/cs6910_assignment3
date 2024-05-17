import torch.nn.functional as F 
import torch.nn as nn
import torch
from dataclasses import dataclass
from torch.nn import Module
import random

# """
# reference:
#         https://discuss.pytorch.org/t/how-to-use-dataclass-with-pytorch/53444/10

# """

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(unsafe_hash=True)
class Encoder(nn.Module):
    """
    A class representing an Encoder module for a sequence-to-sequence model.

    Attributes:
        input_dim (int): The dimensionality of the input data.
        embedded_size (int): The size of the embedded representation.
        hidden_dim (int): The size of the hidden state of the RNN.
        num_layers (int): The number of layers in the RNN.
        bidirectional (bool): Whether the RNN is bidirectional or not.
        cell_type (str): The type of RNN cell, e.g., 'LSTM', 'GRU'.
        dp (float): Dropout probability.

    Example:
        To create an Encoder object:
        
        encoder = Encoder(input_dim=100, embedded_size=50, hidden_dim=128,
                          num_layers=2, bidirectional=True, cell_type='LSTM', dp=0.2)
    
    
    
    """
    input_dim: int
    embedded_size: int
    hidden_dim: int
    num_layers: int
    bidirectional: bool
    cell_type: str
    dp: float

    def __post_init__(self):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(self.dp)

        self.fc_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fc_c = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.dir = 2 if self.bidirectional else 1

        self.embedding = nn.Embedding(self.input_dim, self.embedded_size)

        if self.cell_type == 'RNN':
            self.rnn = nn.RNN(self.embedded_size, self.hidden_dim, self.num_layers, bidirectional=self.bidirectional)
        elif self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(self.embedded_size, self.hidden_dim, self.num_layers, bidirectional=self.bidirectional)
        elif self.cell_type == 'GRU':
            self.rnn = nn.GRU(self.embedded_size, self.hidden_dim, self.num_layers, bidirectional=self.bidirectional)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        if self.bidirectional:
            if self.cell_type == 'LSTM':
                output, (hidden, cell) = self.rnn(embedded)
                hidden = self.fc_h(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
                cell = self.fc_c(torch.cat((cell[0:1], cell[1:2]), dim=2))
                return output, (hidden, cell)
            else:
                output, hidden = self.rnn(embedded)
                hidden = self.fc_h(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
                return output, hidden
        else:
            if self.cell_type == 'LSTM':
                output, (hidden, cell) = self.rnn(embedded)
                return output, (hidden, cell)
            else:
                output, hidden = self.rnn(embedded)
                return output, hidden

    

@dataclass(unsafe_hash=True)
class Decoder(nn.Module):
    """
    A class representing a Decoder module for a sequence-to-sequence model.

    Attributes:
        output_dim (int): The dimensionality of the output data.
        embedded_size (int): The size of the embedded representation.
        hidden_dim (int): The size of the hidden state of the RNN.
        num_layers (int): The number of layers in the RNN.
        bidirectional (bool): Whether the RNN is bidirectional or not.
        cell_type (str): The type of RNN cell, e.g., 'LSTM', 'GRU'.
        dp (float): Dropout probability.

    Example:
        To create a Decoder object:
        
        decoder = Decoder(output_dim=100, embedded_size=50, hidden_dim=128,
                          num_layers=2, bidirectional=True, cell_type='LSTM', dp=0.2)
        
    
    """
    output_dim: int
    embedded_size: int
    hidden_dim: int
    num_layers: int
    bidirectional: bool
    cell_type: str
    dp: float

    def __post_init__(self):
        super(Decoder, self).__init__()
        self.dir = 2 if self.bidirectional else 1

        self.embedding = nn.Embedding(self.output_dim, self.embedded_size)

        if self.cell_type == 'RNN':
            self.rnn = nn.RNN((self.hidden_dim * self.dir) + self.embedded_size, self.hidden_dim, self.num_layers)
        elif self.cell_type == 'LSTM':
            self.rnn = nn.LSTM((self.hidden_dim * self.dir) + self.embedded_size, self.hidden_dim, self.num_layers)
        elif self.cell_type == 'GRU':
            self.rnn = nn.GRU((self.hidden_dim * self.dir) + self.embedded_size, self.hidden_dim, self.num_layers)
        else:
            raise ValueError("Invalid cell type. Choose 'rnn', 'lstm', or 'gru'.")

        self.energy = nn.Linear((self.hidden_dim * (self.dir + 1)), 1)
        self.dropout = nn.Dropout(self.dp)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input, encoder_states, hidden):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden[0].repeat(sequence_length, 1, 1)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        attention = self.softmax(energy)
        attention = attention.permute(1, 2, 0)
        encoder_states = encoder_states.permute(1, 0, 2)

        context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)

        rnn_input = torch.cat((context_vector, embedded), dim=2)

        output, hidden = self.rnn(rnn_input, hidden)

        output = self.fc_out(output)
        output = output.squeeze(0)

        return output, hidden


class Seq2Seq(nn.Module):
    """
    A class representing a Sequence-to-Sequence model.

    This model typically consists of an encoder and a decoder.

    Attributes:
        encoder: The encoder module.
        decoder: The decoder module.
        cell_type (str): The type of RNN cell used in both encoder and decoder.
        bidirectional (bool): Whether the RNNs in both encoder and decoder are bidirectional.

    Example:
        To create a Seq2Seq object:
    
        encoder = Encoder(input_dim=100, embedded_size=50, hidden_dim=128,
                          num_layers=2, bidirectional=True, cell_type='LSTM', dp=0.2)

        decoder = Decoder(output_dim=100, embedded_size=50, hidden_dim=128,
                          num_layers=2, bidirectional=True, cell_type='LSTM', dp=0.2)
                          
        seq2seq_model = Seq2Seq(encoder=encoder, decoder=decoder, cell_type='LSTM', bidirectional=True)

    
    """

    def __init__(self, encoder, decoder,cell_type,bidirectional):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cell_type=cell_type
        self.bidirectional=bidirectional
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]# Get the batch size from the target sequence
        max_len = trg.shape[0]  # Get the maximum length of the target sequence
        trg_vocab_size = self.decoder.output_dim  # Get the vocabulary size of the decoder
        #print(batch_size)
        #print(max_len)
        #print(trg_vocab_size)

        # Tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(device)
        
        # Pass the source sequence through the encoder
        encoder_states, encoder_hidden = self.encoder(src)
        #print("encoder hidden shape",encoder_hidden.shape)
        
        # Set the first decoder input as the first target token
        decoder_input = trg[0]
        #print("decoder input shape",decoder_input.shape)
        
        for t in range(1,max_len ):
             # Pass the decoder input and encoder states through the decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input,encoder_states,encoder_hidden)

            outputs[t] = decoder_output

            # Determine the next decoder input based on teacher forcing ratio
            # Either use the true target token or the predicted token from the previous time step
 
            max_pr=decoder_output.argmax(1)
            decoder_input=trg[t] if random.random()<teacher_forcing_ratio else max_pr

        return outputs