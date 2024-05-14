import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import build_vocab_from_iterator ,Vocab
from torchtext.utils import download_from_url, extract_archive
import io
import pandas as pd
from typing import Set

# url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
# train_urls = ('train.de.gz', 'train.en.gz')
# val_urls = ('val.de.gz', 'val.en.gz')
# test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

# train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
# val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
# test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

# de_tokenizer = get_tokenizer('spacy', language='de')
# en_tokenizer = get_tokenizer('spacy', language='en')


def load(dataset_path):

    df = pd.read_csv(dataset_path)
    
    input = df.iloc[:,0].to_list()   
    output = df.iloc[:,1].to_list()


    return input ,output

def word_to_char(self,word):

    characters: Set[str] = set()
    words_len = []

    for words in word:
        words_len.append(len(words))
        for char in words:
            if char not in characters:
                characters.add(char)

                # print(char)
    
    characters = sorted(list(characters))
    max_word_len = max(words_len)

    return characters #, max_word_len

def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


if __name__ == '__main__':

    examples = ['സ്ഥലമാണ്']

    vocab = build_vocab_from_iterator(examples)
    print(vocab.get_stoi())

