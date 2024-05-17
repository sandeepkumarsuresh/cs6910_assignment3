import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def indices_to_string(trg, t_idx_to_char):
    """Converts a batch of indices to strings using the given index-to-char mapping
    Args:
    trg(Tensor):encoder words of size batch_size x sequence length
    t_idx_to_char(Dict.): index to char mapping

    Returns: Strings
    
    """
    strings = []
    bs=trg.shape[0]
    sq=trg.shape[1]
    for i in range(bs):
      chars = []
      for j in range(sq):
        if trg[i,j].item() in t_idx_to_char:
          char = t_idx_to_char[trg[i,j].item()]
          chars.append(char)
      string = ''.join(chars)
      strings.append(string)
    return strings

def calculate_word_level_accuracy(model,t_idx_to_char,data_loader, criterion):
    model.eval() # Putting the model in evaluation mode. i.e. turning off certain functions
    num_correct = 0
    num_total = 0
    epoch_loss = 0

    with torch.no_grad(): # Turning off gradient
        for batch_idx, (src, trg, src_len, trg_len) in enumerate(data_loader):
            # Convert target indices to string for comparison
            string_trg=indices_to_string(trg,t_idx_to_char)
            
            # Move tensors to the device
            src = src.permute(1, 0)
            trg = trg.permute(1, 0)
            src = src.to(device)
            trg = trg.to(device)
            # Perform forward pass through the model
            output = model(src, trg, 0)
            # turn off teacher forcing
            output = output[1:].reshape(-1, output.shape[2])
            #print("op after ",output.shape) # exclude the start-of-sequence token

            trg = trg[1:].reshape(-1) # exclude the start-of-sequence token
            #print("trg after reshape",trg.shape)
            
            # Calculate the loss
            output = output.to(device)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
            batch_size = trg_len.shape[0]
            #print("bs", batch_size)
            seq_length = int(trg.numel() / batch_size)
            

            # Convert the output to predicted characters
            predicted_indices = torch.argmax(output, dim=1)
            predicted_indices = predicted_indices.reshape(seq_length,-1)
            predicted_indices = predicted_indices.permute(1, 0)
            # Convert predicted indices to strings
            string_pred=indices_to_string(predicted_indices,t_idx_to_char)
            #print(string_pred)
            #print(string_trg)
            
            for i in range(batch_size):
                num_total+=1
                # Compare the predicted string with the target string
                if string_pred[i][:len(string_trg[i])] == string_trg[i]:
                    num_correct+=1

    print("Total",num_total)
    print("Correct",num_correct)
    # Calculate word-level accuracy and average loss
    return (num_correct /num_total) * 100, (epoch_loss/(len(data_loader)))
