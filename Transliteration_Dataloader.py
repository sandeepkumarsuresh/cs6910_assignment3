import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import build_vocab_from_iterator ,Vocab
from torchtext.utils import download_from_url, extract_archive
import io
import pandas as pd
from typing import Set
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset, DataLoader

# url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
# train_urls = ('train.de.gz', 'train.en.gz')
# val_urls = ('val.de.gz', 'val.en.gz')
# test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

# train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
# val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
# test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

# de_tokenizer = get_tokenizer('spacy', language='de')
# en_tokenizer = get_tokenizer('spacy', language='en')


@dataclass
class Transliteration_Dataloader(Dataset):
    file_path:str
    source_lang:str
    target_lang:str

    source_vocab:Any
    target_vocab:Any

    target_char_mapping:Any

    def __post_init__(self):
        self.data_read = pd.read_csv(self.file_path,header = None , names=[self.source_lang,self.target_lang] )

        self.max_source_length = max([len(word) for word in self.data_read[self.source_lang].tolist()]) + 1
        self.max_target_length = max([len(word) for word in self.data_read[self.target_lang].tolist()]) + 1

    def do_one_hot(self, word, char_to_idx):
        num_char = len(char_to_idx)
        max_len = self.max_target_length
        one_hot = torch.zeros((max_len, num_char))
        for i, char in enumerate(word):
            char_idx = char_to_idx.get(char, char_to_idx.get('<unk>'))
            one_hot[i][char_idx] = 1
        return one_hot
    

    ## Modifying the __len__ and the __getitem__ method for the dataloader ##


    def __len__(self):
        return len(self.data_read)

    def __getitem__(self, idx):
        source_data = self.data_read.iloc[idx][self.source_lang]
        target_data = self.data_read.iloc[idx][self.target_lang]

        
        source_vocab_ind = [self.source_vocab.get(char, self.source_vocab.get('<unk>')) for char in source_data]
        target_vocab_ind = [self.target_vocab.get(char, self.target_vocab.get('<unk>')) for char in target_data]
        source_vocab_ind.insert(0, 0)  # Assuming start-of-word token index is 0
        target_vocab_ind.insert(0, 0)  # Assuming start-of-word token index is 0

        source_vocab_len = len(source_vocab_ind)
        target_vocab_len = len(target_vocab_ind)

        pad_source = [self.source_vocab.get('<pad>')] * (self.max_source_length - source_vocab_len)
        pad_target = [self.target_vocab.get('<pad>')] * (self.max_target_length - target_vocab_len)

        source_vocab_ind.extend(pad_source)
        target_vocab_ind.extend(pad_target)

        source_vocab_ind = torch.LongTensor(source_vocab_ind)
        target_vocab_ind = torch.LongTensor(target_vocab_ind)

        return source_vocab_ind, target_vocab_ind, source_vocab_len, target_vocab_len