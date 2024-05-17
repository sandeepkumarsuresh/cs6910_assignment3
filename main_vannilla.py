from DataPreprocess import Build_Vocabulary
from Transliteration_Dataloader import Transliteration_Dataloader
from Load_batch import Data_Load_Batch
from Utility_func import *
from train import Train
from Model import *
# from Model_Attention import *
# from torch import nn
from torch import optim
import torch.nn as nn
import Config
import wandb
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
    NUM_LAYERS = args.num_layers
    EMBEDDING_SIZE = args.embedding_size
    HIDDEN_LAYER = args.hidden_size
    CELL = args.cell_type
    TEACHER_FORCING = args.teacher_forcing
    DROPOUT = args.dropout
    BIDIRECTIONAL = args.bidirectional
    OPTIMIZER = args.optimizer
    LEARNING_RATE = args.learning_rate


    encoder = Encoder(input_dim=INPUT_DIMENSION,
                      embedding_size=EMBEDDING_SIZE,
                      hidden_dim=HIDDEN_LAYER,
                      num_layers=NUM_LAYERS,
                      bidirectional=BIDIRECTIONAL,
                      cell_type=CELL,
                      dropout_value=DROPOUT).to(device)
    
    decoder = Decoder(output_dim=OUTPUT_DIMENSION,
                      embedding_size=EMBEDDING_SIZE,
                      hidden_dim=HIDDEN_LAYER,
                      num_layers=NUM_LAYERS,
                      bidirectional=BIDIRECTIONAL,
                      cell_type=CELL,
                      dropout_value=DROPOUT).to(device)





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
    model_train.train_vannila(tar_idx_to_char)
    # model_train.train_attention(tar_idx_to_char)



if __name__ =='__main__':
	args = Config.parseArguments()
	main(args)





"""

sweep_configuration = {
    'method': 'bayes', #grid, random
    'metric': {
    'name': 'val_acc',
    'goal': 'maximize'   
    },
    'parameters': {
        'num_layers': {
            'values': [3,4,5]
        },
        'hidden_size': {
            'values': [256,512]
        },
        'embedding_size': {
            'values': [256,512]
        },
        'batch_size': {
            'values': [32,64,128]
        },
        'dropout':{
            'values': [0,0.1,0.2,0.5]
        },
        'bidirectional':{
            'values':[True,False]
        },
        'cell_type':{
            'values':["GRU", "LSTM", "RNN"]
        },
        'epochs':{
            'values':[5,10]
        },
        # 'beam_size':{
        #     'values':[3, 4]
        #},
        'learning_rate':{
            'values':[0.001,0.0001]
        },
        'optimiser':{
            'values':['adam','nadam']
        },
        'teacher_forcing':{
            "values":[0.2,0.5,0.7]
        }
    }
}
sweep_id = wandb.sweep(sweep_configuration,project='dl_ass3')


from tqdm import tqdm


def do_train():
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    wandb.init()
    config = wandb.config
    run_name = "cell_type:"+str(config.cell_type)+"epochs"+str(config.epochs)+"num_layers:"+str(config.num_layers)+"hidden_size:"+str(config.hidden_size)+"embedding_size:"+str(config.embedding_size) + "dropout:"+str(config.dropout)+"batch_size:"+str(config.batch_size)+"bidirectional:"+str(config.bidirectional)
    print(run_name)
    wandb.run.name = run_name
  
    # Retrieve the hyperparameters from the config
    ct=config.cell_type
    dp = config.dropout
    em=config.embedding_size
    nlayer=config.num_layers
    bs = config.batch_size
    hs=config.hidden_size
    bidir = config.bidirectional
    lr = config.learning_rate
    opt= config.optimiser
    epochs = config.epochs
    tf=config.teacher_forcing
    trg_pad_idx=0

  

    vocab = Build_Vocabulary('Dataset/mal/mal_train.csv', 'src', 'trg')
    source_vocab, target_vocab = vocab.fetch_vocab()

    INPUT_DIM = len(source_vocab)
    OUTPUT_DIM = len(target_vocab)

    batch_loader_and_char_mapper = Data_Load_Batch(batch_size=bs)
    train_loader,test_loader,val_loader, tar_idx_to_char,src_idx_to_char = batch_loader_and_char_mapper.dataloader_and_char_mapping()


  # Instantiate the Encoder and Decoder models
    encoder = Encoder(INPUT_DIM,em,hs,nlayer,bidir,ct,dp).to(device)
    decoder = Decoder(OUTPUT_DIM,em,hs,nlayer,bidir,ct,dp).to(device)

    model = Seq2Seq(encoder,decoder,ct,bidir).to(device)
 
  # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()      
    if opt == "adam":
          optimizer = optim.Adam(model.parameters(),lr=lr)
    elif opt == "nadam":
          optimizer= optim.NAdam(model.parameters(),lr=lr)
  
  # Train Network
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        model.train()

        for batch_idx, (src, trg, src_len, trg_len) in enumerate(train_loader):
            src = src.permute(1, 0)  # swapping the dimensions of src tensor
            trg = trg.permute(1, 0)  # swapping the dimensions of trg tensor

            src = src.to(device)
            trg = trg.to(device)
            optimizer.zero_grad()
            output = model(src,trg,tf)


            output = output[1:].reshape(-1, output.shape[2])

            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            # if batch_idx % 1000 == 0:
            #     print(f"Epoch: {epoch}, Batch: {batch_idx} , Training..")
        
        train_acc ,train_loss= calculate_word_level_accuracy(model, tar_idx_to_char,train_loader,criterion)
        val_acc,val_loss = calculate_word_level_accuracy(model,tar_idx_to_char, val_loader, criterion)
        # test_acc,test_loss = calculate_word_level_accuracy(model,tar_idx_to_char, test_loader, criterion)
     
        print(f"Epoch: {epoch}, Loss: {epoch_loss / len(train_loader)}, Train Acc: {train_acc}, Val Acc: {val_acc}")

            
        wandb.log({'train_loss': train_loss, 'train_acc': train_acc,'val_acc': val_acc,'train_loss': train_loss,'val_loss': val_loss})
 
    wandb.agent(sweep_id ,function=do_train,count=100)
    wandb.finish()
"""


  
    
