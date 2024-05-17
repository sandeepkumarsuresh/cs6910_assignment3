import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
from Utility_func import calculate_word_level_accuracy
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




@dataclass
class Train:
    """
    A class to represent the training process of a machine learning model.

    Attributes:
    - epoch (int): The number of epochs for training.
    - model (object): The machine learning model to be trained.
    - optimizer (object): The optimizer used for updating model parameters during training.
    - loss_func (object): The loss function used for calculating the training loss.
    - train_loader (object): The data loader for the training dataset.
    - val_loader (object): The data loader for the validation dataset.
    - teacher_forcing (bool): Whether to use teacher forcing during training.

    Methods:
    - train_vanilla(self, t_idx_to_char): Trains the model using vanilla sequence-to-sequence training.
    - train_attention(self,t_idx_to_char): Trains the model using attention sequence-to-sequence training.
    """

    epoch: int
    model: object
    optimizer: object
    loss_func: object
    train_loader: object
    val_loader: object
    teacher_forcing: float


    def train_vannila(self,t_idx_to_char):
        # Train the model
        for epoch in tqdm(range(self.epoch)):
            epoch_loss = 0
            
            # Setting the Model to Training Mode
            self.model.train()

            for batch_idx, (src, trg, src_len, trg_len) in enumerate(self.train_loader):

                src = src.permute(1, 0)  
                trg = trg.permute(1, 0)  

                src = src.to(device)
                trg = trg.to(device)
                
                self.optimizer.zero_grad()
                
                output= self.model(src, trg,self.teacher_forcing)

                
                output = output[1:].reshape(-1, output.shape[2])

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
        
        # Save best model
        best_model_path = 'best_model_vanillaSeq2Seq.pth'
        torch.save(self.model.state_dict(), best_model_path)
        print(f"Best model saved to {best_model_path}")


    def train_attention(self,t_idx_to_char):

        # Train the model
        for epoch in tqdm(range(self.epoch)):
            epoch_loss = 0
            
            # Setting the Model to Training Mode
            self.model.train()

            for batch_idx, (src, trg, src_len, trg_len) in enumerate(self.train_loader):

                src = src.permute(1, 0) 
                trg = trg.permute(1, 0) 

                src = src.to(device)
                trg = trg.to(device)
                
                self.optimizer.zero_grad()
                output = self.model(src, trg,self.teacher_forcing)
                
                
                output = output[1:].reshape(-1, output.shape[2])

                trg = trg[1:].reshape(-1)
                
                loss = self.loss_func(output, trg)
                loss.backward()
                
                self.optimizer.step()
                
                epoch_loss += (loss.item())
                
                if batch_idx % 1000 == 0:
                    print(f" Training ... Batch: {batch_idx},")

            # Calculate word-level accuracy after every epoch
            val_acc,val_loss = calculate_word_level_accuracy(self.model,t_idx_to_char,self.val_loader,self.loss_func)
            
            print(f"Epoch: {epoch}, Loss: {epoch_loss / (len(self.train_loader))}, Val Acc: {val_acc}, Val loss: {val_loss}")
