import pandas as pd
from dataclasses import dataclass
from typing import Set
import unicodedata
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
SOS_token = 0
EOS_token = 1

@dataclass
class Build_Vocabulary:

    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "<SOS>", 1: "<EOS>",2:"<PAD>"}
        self.n_chars = 3  # Count SOS ,EOS,PAD

    def addSentance(self, sentance):
        for char in sentance:
            self.addChar(char)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

def unicodeToAscii(self,s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(self,s):
    s = self.unicodeToAscii(s.lower().strip())
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # Optionally remove unwanted characters
    return s

def readLangs(lang1, lang2, file, reverse=False):
    print("Reading lines...")
    '''df = pd.read_csv(file, header=None, usecols=[0, 1])  # Only read the first two columns

    # Normalize strings and prepare pairs
    pairs = [[normalizeString(str(x)) for x in row] for index, row in df.iterrows()]'''
    df = pd.read_csv(file, header=None, usecols=[0, 1])
    # Convert the dataframe rows to a list of pairs
    pairs = df.values.tolist()
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Build_Vocabulary(lang2)
        output_lang = Build_Vocabulary(lang1)
    else:
        input_lang = Build_Vocabulary(lang1)
        output_lang = Build_Vocabulary(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, file, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, file, reverse)
    print(f"Read {len(pairs)} sentence pairs")
    for pair in pairs:
        
        input_lang.addSentance(pair[0])
        output_lang.addSentance(pair[1])

    print("Counted chars:")
    print(input_lang.name, input_lang.n_chars, input_lang.index2char)
    print(output_lang.name, output_lang.n_chars, output_lang.index2char)
    return input_lang, output_lang, pairs



file_path = 'Dataset/mal/mal_train.csv' # Update this path
input_lang, output_lang, pairs = prepareData('eng', 'mal', file_path, reverse=False)
print(random.choice(pairs))


MAX_LENGTH = max(max(len(pair[0]), len(pair[1])) for pair in pairs) + 1


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

def indexesFromSentence(lang, sentence):
    return [lang.char2index[char] for char in sentence]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader(batch_size):
    file='Dataset/mal/mal_train.csv'
    
    input_lang, output_lang, pairs = prepareData('eng', 'mal', file)

    n = len(pairs)
    print('fthe no of pairs{n}')
    MAX_LENGTH = max(max(len(pair[0]), len(pair[1])) for pair in pairs) + 1
    print(f'max length is{MAX_LENGTH}')
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    print(f'shape of input and output is {input_ids.shape,target_ids.shape}')

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        
        tgt_ids = indexesFromSentence(output_lang, tgt)
        
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids
        # print(f'the input pair and char to index inp_ids{pairs[0],inp_ids}')

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data
        # print(input_tensor)
        input_tensor_cpu = input_tensor.cpu()

        # Print and check characters for each sequence in the batch
        for sequence in input_tensor_cpu:
            check_chars = [input_lang.index2char[idx.item()] if idx != EOS_token else '<EOS>' for idx in sequence]
            check_word = ''.join(check_chars)
            # print(check_word) 

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    print(total_loss / len(dataloader))
        # decoded_output , _= evaluate(encoder, decoder, input_tensor, input_lang, output_lang)
        # print("Evaluated Output:", decoded_output)

    return total_loss / len(dataloader)


import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


from tqdm import tqdm 

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden,_ = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_chars = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_chars.append('<EOS>')
                break
            decoded_chars.append(output_lang.index2char[idx.item()])
    return decoded_chars

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    print("Training Started")
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in tqdm(range(1, n_epochs + 1)):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        


    # showPlot(plot_losses)
device = 'cpu'
hidden_size = 128
batch_size = 32

input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

encoder = EncoderRNN(input_lang.n_chars, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, output_lang.n_chars).to(device)

# train(train_dataloader, encoder, decoder, 2, print_every=5, plot_every=5)



def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

encoder.eval()
decoder.eval()
evaluateRandomly(encoder, decoder)


# @dataclass
# class LoadDataset:
#     """
#     The primary function of the class is to load the dataset and return
#     it as an input-output pair
    
#     """
    
    
#     dataset_path :str

#     # Declaring eos,sos and pad tokens
#     EOS : str = '<eos>'
#     SOS : str = '<sos>'
#     PAD : str = 'pad'
    

#     def load(self):

        
#         df = pd.read_csv(self.dataset_path)
        
#         self.input = df.iloc[:,0].to_list()  
#         self.output = df.iloc[:,1].to_list()


#         return self.input ,self.output

#     def word_to_char(self,word):
#         """
#         Returns a unique character dictionary

#         Parameters
#         ----------
#         word: list of words

#         Returns
#         -------
#         characters : returns a unique set of characters
#         max_word_len : returns the maximum word length
        
#         """

#         # Creating a unique set
#         characters: Set[str] = set()
#         words_len = []

#         for words in word:
#             # print(words)
#             words_len.append(len(words))
#             for char in words:
#                 if char not in characters:
#                     characters.add(char)

#                     # print(char)
        
#         characters = sorted(list(characters))
#         max_word_len = max(words_len)

#         return characters , max_word_len
    
#     def char_mapping(self,input_char):
#         """
#         Returns the total number of characters and character to index map

#         Parameters
#         ----------
#         input_char: list of unique characters

#         Returns
#         -------
#         num_char : The length of the character sequence
#         char_index_map : mapping of character to the index
        


#         """
#         # Adding Special Tokens to the Characters
#         # EOS and SOS Tokens
#         # https://datascience.stackexchange.com/questions/26947/why-do-we-need-to-add-start-s-end-s-symbols-when-using-recurrent-neural-n

#         char_index_map = {}


#         char_index_map[0] = self.SOS
#         char_index_map[1] = self.EOS
#         char_index_map[2] = self.PAD


#         num_char = len(input_char)

#         for i,char in enumerate(input_char):
#             char_index_map[i+3] = char
        

#         return num_char,char_index_map

# @dataclass
# class Tokenizer:

#     EOS : str = '<eos>'
#     SOS : str = '<sos>'
#     PAD : str = '<pad>'

#     # target_characters: Set[str] = field(default_factory=set)



    


#     def tensor_mapping(self,char,num_tokens,token_index_map):
#         """
#         This function converts one character into a tensor 

#         This function is convert to tensor for a single character
#         """
        
#         tensor = torch.zeros(1,num_tokens,dtype=torch.long)

#         for key,value in token_index_map.items():
#             # print(value)
#             if value == char:
#                 index = key
#                 break

#         else:
#             raise ValueError("Character might be uppercase . Check language input given")
            
#         tensor[0][index] = 1
#         # print(tensor)

#         return tensor
    

#     def convert_tensor_word(self,word,num_tokens,token_index_map):
#         """
#         Convert a word into a tensor size of the form given below

#         <n_letters_in_a_word x 1 x num_tokens>
#         """
        
#         tensor = torch.zeros(len(word),1,num_tokens,dtype=torch.long)

#         for i , char in enumerate(word):

#             for key,value in token_index_map.items():
#                 # print(value)
#                 if value == char:
#                     index = key
#                     break
#             else:
#                 raise ValueError("Character might be uppercase . Check language input given")
                
#             tensor[i][0][index] = 1
        
#         # print(tensor.shape)

#         return tensor


# @dataclass
# class TransliterationDataset:
#     filepath : str = 'Dataset/mal/mal_train.csv'


#     def __post_init__(self):
#         self.transliterations = pd.read_csv(self.filepath,header=None,usecols=0)
#         sr


# if __name__ == '__main__':

#     path_to_dataset = 'Dataset/mal/mal_train.csv'

#     x = LoadDataset(path_to_dataset)
#     a,b = x.load()
#     # print(a)

#     y = Tokenizer()

#     c,_ = x.word_to_char(a)
#     print(c)

#     d,e = x.char_mapping(c)
#     print(d)
#     print(e)

#     # f = y.convert_tensor_word(word='hello',num_tokens=d,token_index_map=e)

