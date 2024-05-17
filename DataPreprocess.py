import pandas as pd
from dataclasses import dataclass
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler

@dataclass
class Build_Vocabulary:

    """
        A class to build vocabulary from source and target language texts.

    Attributes:
        file_path (str): The path to the file containing source and target language texts.
        src_lang (str): The source language.
        trg_lang (str): The target language.
        translations (pd.DataFrame, optional): A DataFrame containing translations, default is None.
        src_lang_vocab (dict, optional): A dictionary containing the vocabulary for the source language, default is None.
        target_lang_vocab (dict, optional): A dictionary containing the vocabulary for the target language, default is None.
        t_char_to_idx (dict, optional): A dictionary mapping characters to indices for the target language, default is None.
        t_idx_to_char (dict, optional): A dictionary mapping indices to characters for the target language, default is None.
        s_char_to_idx (dict, optional): A dictionary mapping characters to indices for the source language, default is None.

    Example:
        To create a Build_Vocabulary object:
        
        vocab_builder = Build_Vocabulary(file_path='data.txt', src_lang='en', trg_lang='fr')
        
    
    
    """

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

        """
        Post Initialization of the variable 
        """

        self.translations = pd.read_csv(self.file_path, header=None, names=[self.src_lang, self.trg_lang])
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

        self.s_char_to_idx = {char: idx + 3 for idx, char in enumerate(source_characters)}
        self.s_char_to_idx['<unk>'] = 2
        self.s_idx_to_char = {idx: char for char, idx in self.s_char_to_idx.items()}
        
        self.t_char_to_idx = {char: idx + 3 for idx, char in enumerate(target_characters)}
        self.t_char_to_idx['<unk>'] = 2
        self.t_idx_to_char = {idx: char for char, idx in self.t_char_to_idx.items()}
        
    def fetch_vocab(self):
        """
        Returns Vocabulary for Source and Target Languages
        """
        return self.source_lang_vocab, self.target_lang_vocab

    def fetch_char_index_mapping(self):
        """
        Returns Character to Indices and Indices to Character Mapping for Source and Target Languages
        """
        return  self.t_char_to_idx, self.t_idx_to_char, self.s_char_to_idx, self.s_idx_to_char
