import torch.nn.functional as F 
import torch.nn as nn
# import torch.nn as nn
import torch
from dataclasses import dataclass
from torch.nn import Module
# from torch.nn import Embedding


"""
reference:
        https://discuss.pytorch.org/t/how-to-use-dataclass-with-pytorch/53444/10

"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



@dataclass(unsafe_hash=True)
class Encoder(Module):
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
    input_dim:int
    embedding_size:int
    hidden_dim:int
    num_layers:int
    bidirectional:str
    cell_type:str
    dropout_value:float
    
    def __post_init__(self):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(self.input_dim,self.embedding_size)
        # print('---Inside Encoder---')
        # print(type(self.cell_type))
        # self.bidirectional = self.bidirectional.lower() == 'true'

        if self.bidirectional:
            self.dir=2
        else:
            self.dir=1  
        # Create the recurrent layer based on the specified cell type
        if (self.cell_type) == 'RNN':
              self.rnn = nn.RNN(input_size=self.embedding_size, hidden_size=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout_value,bidirectional=self.bidirectional)
        elif self.cell_type == 'LSTM':
              self.rnn = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout_value,bidirectional=self.bidirectional)
        elif self.cell_type == 'GRU':
              self.rnn = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout_value,bidirectional=self.bidirectional)
    
    def forward(self, source):
        # Apply dropout to the embedded input
        dropout = nn.Dropout(self.dropout_value)
        embedded = dropout(self.embedding(source))
        # print('embed',type(embedded))
        # If the cell type is LSTM, return both the output and the hidden and cell states in a single tuple
        if self.cell_type == 'LSTM':
            output, (hidden, cell) = self.rnn(embedded)
            return output, (hidden, cell)

        else:
            # For other cell types (RNN, GRU), return the output and the hidden state
            output, hidden = self.rnn(embedded)
            return output,hidden


        
@dataclass(unsafe_hash=True)
class Decoder(Module):
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

    output_dim:int
    embedding_size:int
    hidden_dim:int
    num_layers:int
    bidirectional:str
    cell_type:str
    dropout_value:float

    def __post_init__(self):
        super(Decoder, self).__init__()

        # self.bidirectional = self.bidirectional.lower() 

        if self.bidirectional:
            self.dir=2
        else:
            self.dir=1  

        # Create an embedding layer
        # print("Inside Decoder")
        # print(self.output_dim,self.embedding_size)
        self.embedding = nn.Embedding(self.output_dim,self.embedding_size)
        # Create the recurrent layer based on the specified cell type
        if self.cell_type == 'RNN':
            self.rnn = nn.RNN(self.embedding_size, self.hidden_dim, self.num_layers,dropout=self.dropout_value)
        elif self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(self.embedding_size, self.hidden_dim, self.num_layers,dropout=self.dropout_value)
        elif self.cell_type == 'GRU':
            self.rnn = nn.GRU(self.embedding_size, self.hidden_dim, self.num_layers,dropout=self.dropout_value)

        # Create the output fully connected layer
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, input, hidden):
        dropout = nn.Dropout(self.dropout_value)

        embedded = dropout(self.embedding(input))
        # Pass the embedded input and hidden state through the decoder RNN
        output, hidden = self.rnn(embedded, hidden)
        # Pass the decoder output through the fully connected layer
        output = self.fc_out(output)
        # Apply log softmax activation to the output
        output = F.log_softmax(output, dim=1)

        return output, hidden


# @dataclass(unsafe_hash=True)
class Seq2Seq(Module):

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


        batch_size = trg.shape[1]
        #print(batch_size)
        max_len = trg.shape[0]
        #print(max_len)
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(device)

        # print((src).shape)
        
        encoder_output, encoder_hidden = self.encoder(src)
        # print('inside seq2seq',encoder_hidden.shape)
        # print(self.encoder.num_layers)

        #print("encoder hidden shape",encoder_hidden.shape)
        # Concatenate the last hidden state of the encoder from both directions
        # print(self.bidirectional)
        if self.bidirectional:
            if self.cell_type=='LSTM':
                hidden_concat = torch.add(encoder_hidden[0][0:self.encoder.num_layers,:,:], encoder_hidden[1][0:self.encoder.num_layers,:,:])/2
                cell_concat = torch.add(encoder_hidden[0][self.encoder.num_layers:,:,:], encoder_hidden[1][self.encoder.num_layers:,:,:])/2
                hidden_concat = (hidden_concat, cell_concat)

            else:
                # print('inside the self.bidirectional loop')
                hidden_concat = torch.add(encoder_hidden[0:self.encoder.num_layers,:,:], encoder_hidden[self.encoder.num_layers:,:,:])/2
        else:
            hidden_concat= encoder_hidden
        
        decoder_hidden = hidden_concat
        # Initialize decoder input with the start token
        decoder_input = (trg[0,:]).unsqueeze(0)
        #print("decoder input shape",decoder_input.shape)
        
        for t in range(1,trg.shape[0] ):

            # Pass the decoder input and hidden state through the decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Store the decoder output in the outputs tensor
            outputs[t] = decoder_output
            # Determine the next decoder input using teacher forcing or predicted output
            max_pr, idx=torch.max(decoder_output,dim=2)
            #print("trg shape",trg.shape)
            idx=idx.view(trg.shape[1])
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            if teacher_force:
                decoder_input= trg[t,:].unsqueeze(0)
            else:
                decoder_input= idx.unsqueeze(0)

         # Pass the last decoder input and hidden state through the decoder
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        
        # Store the final decoder output in the outputs tensor
        #outputs[-1] = decoder_output
        return outputs