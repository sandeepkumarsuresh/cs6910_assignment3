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

@dataclass
class Transliteration_Dataloader(Dataset):
    """
    A class representing a DataLoader for transliteration tasks.

    Attributes:
        file_path (str): The path to the file containing transliteration data.
        source_lang (str): The source language of the transliteration data.
        target_lang (str): The target language of the transliteration data.
        source_vocab (Any): Vocabulary or mapping for the source language.
        target_vocab (Any): Vocabulary or mapping for the target language.
        target_char_mapping (Any): Character mapping specific to the target language.

    Example:
        To create a Transliteration_Dataloader object:

        dataloader = Transliteration_Dataloader(file_path='data.txt', source_lang='en', target_lang='fr',
                                                source_vocab=source_vocab, target_vocab=target_vocab,
                                                target_char_mapping=target_char_mapping)
    """
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
        """
        Converts the Character into one-hot encodings
        """
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
        source_vocab_ind.insert(0, 0)  
        target_vocab_ind.insert(0, 0)  

        source_vocab_len = len(source_vocab_ind)
        target_vocab_len = len(target_vocab_ind)

        pad_source = [self.source_vocab.get('<pad>')] * (self.max_source_length - source_vocab_len)
        pad_target = [self.target_vocab.get('<pad>')] * (self.max_target_length - target_vocab_len)

        source_vocab_ind.extend(pad_source)
        target_vocab_ind.extend(pad_target)

        source_vocab_ind = torch.LongTensor(source_vocab_ind)
        target_vocab_ind = torch.LongTensor(target_vocab_ind)

        return source_vocab_ind, target_vocab_ind, source_vocab_len, target_vocab_len