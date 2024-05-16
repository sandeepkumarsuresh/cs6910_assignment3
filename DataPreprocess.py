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

@dataclass
class Build_Vocabulary:

    file_path: str
    src_lang: str
    trg_lang: str
    translations: pd.DataFrame = None
    src_lang_vocab: dict = None
    target_lang_vocab: dict = None
    t_char_to_idx: dict = None
    t_idx_to_char: dict = None
    s_char_to_idx: dict = None
    s_idx_to_char: dict = None

    def __post_init__(self):

        self.translations = pd.read_csv(self.file_path, header=None, names=[self.src_lang, self.trg_lang])
        # self.translations.dropna(inplace=True)
        self.source_lang_vocab = {char: i + 3 for i, char in enumerate(sorted(set(''.join(self.translations[self.src_lang].tolist()))))}

        self.target_lang_vocab = {char: i + 3 for i, char in enumerate(sorted(set(''.join(self.translations[self.trg_lang].tolist()))))}
        
        # Add special tokens to the vocabularies.
        self.source_lang_vocab['<'] = 0
        self.source_lang_vocab['<unk>'] = 2
        self.source_lang_vocab['<pad>'] = 1
        
        self.target_lang_vocab['<'] = 0
        self.target_lang_vocab['<unk>'] = 2
        self.target_lang_vocab['<pad>'] = 1
        
        source_characters = sorted(set(''.join(self.translations[self.src_lang])))
        target_characters = sorted(set(''.join(self.translations[self.trg_lang])))

        # Assign an index to each character in the source and target languages
        self.s_char_to_idx = {char: idx + 3 for idx, char in enumerate(source_characters)}
        self.s_char_to_idx['<unk>'] = 2
        self.s_idx_to_char = {idx: char for char, idx in self.s_char_to_idx.items()}
        
        self.t_char_to_idx = {char: idx + 3 for idx, char in enumerate(target_characters)}
        self.t_char_to_idx['<unk>'] = 2
        self.t_idx_to_char = {idx: char for char, idx in self.t_char_to_idx.items()}
        
    def fetch_vocab(self):
        # This function returns the source and target vocabularies, as well as the dictionaries that map characters to integer indexes and vice versa.
        return self.source_lang_vocab, self.target_lang_vocab

    def fetch_char_index_mapping(self):
        return  self.t_char_to_idx, self.t_idx_to_char, self.s_char_to_idx, self.s_idx_to_char
