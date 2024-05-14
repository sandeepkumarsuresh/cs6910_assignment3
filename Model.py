import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# TODO
# Adding number of layers
# Adding dropout
# Adding an embedding value

@dataclass
class Encoder_Config:
    hidden_size:int
    n_layers:int 
    input_size:int = 26  # Total number of letters in English
    embed_size :int = 16

class EncoderRNN(nn.Module):
    def __init__(self, config:Encoder_Config):
        super(EncoderRNN, self).__init__()
        self.config = config
        # self.input_size = input_size
        # self.embedd_size = embedd_size

        self.embedding = nn.Embedding(config.input_size, config.embed_size)
        self.rnn = nn.RNN(input_size=config.embed_size,
                          hidden_size = config.hidden_size,
                          num_layers=config.n_layers,
                          batch_first=True)
        # self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):

        # shape of x : seq_length x N
        print('shape of input',x.shape)
        embedded = self.embedding(x)
        # shape of embedded : seq_length x N x embedding size
        print('embedded',embedded.shape)
        output, hidden = self.rnn(embedded)
        # output, hidden = self.rnn(input.float())
        return output, hidden



    def initHidden(self):
        return torch.zeros(1, self.config.hidden_size)
    

@dataclass
class Decoder_Config:
    input_size:int
    hidden_size:int
    output_size:int
    n_layers:int
    embed_size:int

class DecoderRNN(nn.Module):
    def __init__(self,config:Decoder_Config):
        super(DecoderRNN, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.input_size, config.hidden_size)
        self.rnn = nn.RNN(config.embed_size, config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.output_size)
        self.softmax = nn.LogSoftmax() # dim =1


    def forward(self,input,hidden):

        input = input.unsqueeze(0)
        decoder_embedding = self.embedding(input)
        # activation = F.relu(decoder_embedding)
        output , hidden = self.rnn(decoder_embedding,hidden)
        output = self.out(output)
        output = self.softmax(output)
        return output,hidden
    
@dataclass
class Train_Config:
    # input_tensor:torch.Tensor
    # target_tensor:torch.Tensor
    encoder:torch.nn.Module
    decoder:torch.nn.Module


class Seq2Seq(nn.Module):


    def __init__(self, config:Train_Config):
        self.config = config
        super(Seq2Seq,self).__init__()


    def forward(self,source,target):


        batch_size = source.shape[1]
        target_len = target.shape[0]

        target_vocab_size = 69

        outputs = torch.zeros(target_len, batch_size, target_vocab_size)

        
        enc_output , enc_hidden = self.config.encoder(source)

        x = target[0]

        for t in range(1,target_len):

            output_decoder , hidden = self.config.decoder(x,enc_hidden)

            outputs[t] = output_decoder

            best_guess  = output_decoder.argmax()


            print(output_decoder)

        return outputs

    # def train_seq2seq(self):

    #     hidden = self.encoder.initHidden()

