from DataPreprocess import Build_Vocabulary
from Transliteration_Dataloader import Transliteration_Dataloader
from Load_batch import Data_Load_Batch
from Utility_func import *
from train import Train
from Model_Attention import *
from torch import optim
import torch.nn as nn
import Config

def main(args):
    BATCH_SIZE = args.batch_size

    # Defining all the objects
    vocab = Build_Vocabulary('Dataset/mal/mal_train.csv','src','tar')
    batch_loader_and_char_mapper = Data_Load_Batch(batch_size=BATCH_SIZE)

    # Preparing the Dataset
    source_vocab, target_vocab = vocab.fetch_vocab()

    # Loading the Dataloader
    train_dataloader,test_dataloader,val_dataloader, tar_idx_to_char,src_idx_to_char = batch_loader_and_char_mapper.dataloader_and_char_mapping()



    # Defining the Source and Target Vocab
    INPUT_DIMENSION = len(source_vocab)
    OUTPUT_DIMENSION = len(target_vocab)
    # print(type(OUTPUT_DIMENSION))
    EPOCH = args.epochs
    NUM_LAYERS = 1   # For attention , we have taken layers size = 1
    EMBEDDING_SIZE = args.embedding_size
    HIDDEN_LAYER = args.hidden_size
    CELL = args.cell_type
    TEACHER_FORCING = args.teacher_forcing
    DROPOUT = args.dropout
    BIDIRECTIONAL = args.bidirectional
    OPTIMIZER = args.optimizer
    LEARNING_RATE = args.learning_rate


    encoder = Encoder(input_dim=INPUT_DIMENSION,
                      embedded_size=EMBEDDING_SIZE,
                      hidden_dim=HIDDEN_LAYER,
                      num_layers=NUM_LAYERS,
                      bidirectional=BIDIRECTIONAL,
                      cell_type=CELL,
                      dp=DROPOUT).to(device)
    
    decoder = Decoder(output_dim=OUTPUT_DIMENSION,
                      embedded_size=EMBEDDING_SIZE,
                      hidden_dim=HIDDEN_LAYER,
                      num_layers=NUM_LAYERS,
                      bidirectional=BIDIRECTIONAL,
                      cell_type=CELL,
                      dp=DROPOUT).to(device)

    


    model = Seq2Seq(encoder,
                    decoder,
                    CELL,
                    BIDIRECTIONAL).to(device)


    loss_function = nn.CrossEntropyLoss()

    if OPTIMIZER == "adam":
        optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
    elif OPTIMIZER == "nadam":
        optimizer = optim.NAdam(model.parameters(),lr=LEARNING_RATE)
    
    model_train = Train(epoch=EPOCH,
                        model=model,
                        optimizer=optimizer,
                        loss_func=loss_function,
                        train_loader=train_dataloader,
                        val_loader=val_dataloader,
                        teacher_forcing=TEACHER_FORCING
                        )
    
    # print('tar_id',type(tar_idx_to_char))
    # model_train.train_vannila(tar_idx_to_char)
    model_train.train_attention(tar_idx_to_char)
    
if __name__ =='__main__':
	args = Config.parseArguments()
	main(args)


  
    
