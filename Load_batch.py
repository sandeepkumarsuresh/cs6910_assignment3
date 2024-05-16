from dataclasses import dataclass
from DataPreprocess import Build_Vocabulary
from Transliteration_Dataloader import *

@dataclass
class Data_Load_Batch:

    train_path : str = "/home/sandeep/Documents/IITM/Deep Learning/CS6910_Assignment3_Anuska/Dataset/mal/mal_train.csv"
    val_path : str = "/home/sandeep/Documents/IITM/Deep Learning/CS6910_Assignment3_Anuska/Dataset/mal/mal_valid.csv"
    test_path : str = "/home/sandeep/Documents/IITM/Deep Learning/CS6910_Assignment3_Anuska/Dataset/mal/mal_test.csv"

    batch_size : int = 32 # Setting Default to 32

    def dataloader_and_char_mapping(self):

        data_preprocess = Build_Vocabulary(self.train_path,'src','tar')

        # fetching Vocab
        source_lang_vocab, target_lang_vocab = data_preprocess.fetch_vocab()
       
        # fetching character to index mapping
        tar_char_to_idx, tar_idx_to_char, src_char_to_idx, src_idx_to_char = data_preprocess.fetch_char_index_mapping()

        train_dataset = Transliteration_Dataloader(self.train_path, 'src', 'trg',source_lang_vocab,target_lang_vocab,tar_char_to_idx)
        val_dataset = Transliteration_Dataloader(self.val_path, 'src', 'trg',source_lang_vocab,target_lang_vocab,tar_char_to_idx)
        test_dataset = Transliteration_Dataloader(self.test_path, 'src', 'trg',source_lang_vocab,target_lang_vocab,tar_char_to_idx)
        
        # Create train, validation, and test data loaders
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader,test_dataloader,val_dataloader, tar_idx_to_char,src_idx_to_char
   