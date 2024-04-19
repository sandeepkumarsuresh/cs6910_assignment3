import pandas as pd
from dataclasses import dataclass,field
from typing import Set
import torch

@dataclass
class LoadDataset:
    
    
    dataset_path :str
    input : list
    output : list

    
    def load(self):
        """
        This function loads the dataset and returns it as a list

        """

        df = pd.read_csv(self.dataset_path)
        
        self.input = df.iloc[:,0].to_list()   
        self.output = df.iloc[:,1].to_list()


        return self.input ,self.output


@dataclass
class Tokenizer:
    characters: Set[str] = field(default_factory=set)
    # target_characters: Set[str] = field(default_factory=set)


    def tokenizer(self,word):
        """
        Converting character to unique index
        """

        for words in word:
            for char in words:
                if char not in self.characters:
                    self.characters.add(char)

                    # print(char)
        
        self.characters = sorted(list(self.characters))
        return self.characters
    
    def token_mapping(self,input_char):
        """
        Getting the total number of tokens and
        mapping characters into 
        """

        num_tokens = len(input_char)
        token_index_map = {}

        for i,char in enumerate(input_char):
            token_index_map[i] = char
        

        return num_tokens,token_index_map
        # num_decoder_token = len(target_char)

    def tensor_mapping(self,char,num_tokens,token_index_map):
        """
        This function converts one character into a tensor 

        This function is convert to tensor for a single character
        """
        
        tensor = torch.zeros(1,num_tokens)

        for key,value in token_index_map.items():
            print(value)
            if value == char:
                index = key
                break

        else:
            raise ValueError("Character might be uppercase . Check language input given")
            
        tensor[0][index] = 1
        print(tensor)

        return tensor
    

    def convert_tensor_word(self,word,num_tokens,token_index_map):
        """
        Convert a word into a tensor size of the form given below

        <n_letters_in_a_word x 1 x num_tokens>
        """
        
        tensor = torch.zeros(len(word),1,num_tokens)

        for i , char in enumerate(word):

            for key,value in token_index_map.items():
                print(value)
                if value == char:
                    index = key
                    break
            else:
                raise ValueError("Character might be uppercase . Check language input given")
                
            tensor[i][0][index] = 1
        
        print(tensor.shape)

        return tensor


if __name__ == '__main__':

    path_to_dataset = 'Dataset/mal/mal_train.csv'

    x = LoadDataset(path_to_dataset)
    a,b = x.load()

    y = Tokenizer()

    c = y.tokenizer(a)

    d,e = y.token_mapping(c)

    # print(d)
    # print(e)

    f = y.convert_tensor_word(word='hello',num_tokens=d,token_index_map=e)
