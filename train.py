import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
from Utility_func import calculate_word_level_accuracy
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# @dataclass
# class Train:
#     epoch:int
#     model:nn.Module
#     optimizer:optim.Optimizer
#     loss_func:nn.Module
#     train_loader:DataLoader
#     val_loader:DataLoader
#     teacher_forcing:float

class Train:
    def __init__(self, epoch, model, optimizer, loss_func, train_loader, val_loader, teacher_forcing):
        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.teacher_forcing = teacher_forcing

    def train(self,t_idx_to_char):
        # Train the model
        for epoch in tqdm(range(self.epoch)):
            epoch_loss = 0
            
            # Setting the Model to Training Mode
            self.model.train()

            for batch_idx, (src, trg, src_len, trg_len) in enumerate(self.train_loader):
                # print(batch_idx)
                # print(src.shape)
                # print(trg.shape)
                src = src.permute(1, 0)  # swapping the dimensions of src tensor
                trg = trg.permute(1, 0)  # swapping the dimensions of trg tensor

                src = src.to(device)
                trg = trg.to(device)
                
                self.optimizer.zero_grad()
                # print(type(src),type(trg),type(self.teacher_forcing))
                
                output= self.model(src, trg,self.teacher_forcing)
                # print(output)
                
                # Ignore the first element of the output, which is initialized as all zeros
                # since we use it to store the output for the start-of-sequence token
                #print(output.shape[2])
                
                output = output[1:].reshape(-1, output.shape[2])
                #print(output.shape)
                #print(trg.shape)
                trg = trg[1:].reshape(-1)
                
                loss = self.loss_func(output, trg)
                loss.backward()
                
                self.optimizer.step()
                
                epoch_loss += (loss.item())
                
                if batch_idx % 1000 == 0:
                    print(f"Epoch: {epoch}, Batch: {batch_idx}, Training...")

            # Calculate word-level accuracy after every epoch
            val_acc,val_loss = calculate_word_level_accuracy(self.model,t_idx_to_char,self.val_loader,self.loss_func)
            
            print(f"Epoch: {epoch}, Loss: {epoch_loss / (len(self.train_loader))}, Val Acc: {val_acc}, Val loss: {val_loss}")
            #wandb.log({'epoch': epoch, 'loss': loss.item(), 'test_acc': test_acc,'train_acc': train_acc,'val_acc': val_acc})
            
        
        # Save best model
        best_model_path = 'best_model_vanillaSeq2Seq.pth'
        torch.save(self.model.state_dict(), best_model_path)
        print(f"Best model saved to {best_model_path}")
