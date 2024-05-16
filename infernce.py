import torch
import torch.nn as nn
from Model import Encoder, Decoder, Seq2Seq
from your_data_loader_file import load_data

# Define hyperparameters
INPUT_DIM = 29
OUTPUT_DIM = 67
embedding_size = 512
HIDDEN_DIM = 512
NUM_LAYERS = 3
CELL_TYPE = 'gru'
BATCH_SIZE = 128
TEACHER_FORCING_RATIO = 0.7

dropout = 0.1
bidirectional = True

# Load data and create data loaders
_, _, _, t_idx_to_char, s_idx_to_char = load_data(BATCH_SIZE)

# Instantiate the Encoder and Decoder models
encoder = Encoder(INPUT_DIM, embedding_size, HIDDEN_DIM, NUM_LAYERS, bidirectional, CELL_TYPE, dropout)
decoder = Decoder(OUTPUT_DIM, embedding_size, HIDDEN_DIM, NUM_LAYERS, bidirectional, CELL_TYPE, dropout)

# Instantiate the Seq2Seq model with the Encoder and Decoder models
model = Seq2Seq(encoder, decoder, CELL_TYPE, bidirectional)

# Load the trained model state
model.load_state_dict(torch.load('best_model_vanillaSeq2Seq.pth'))
model.eval()

# Function to perform inference
def inference(sentence):
    with torch.no_grad():
        # Convert input sentence to tensor
        src_tensor = torch.tensor(sentence, dtype=torch.long).unsqueeze(1)  # Add batch dimension
        src_tensor = src_tensor.to(device)

        # Pass input through encoder
        encoder_outputs, hidden = model.encoder(src_tensor)

        # Initialize decoder input with SOS_token
        trg_tensor = torch.tensor([[s_idx_to_char['<sos>']]], device=device)  # SOS_token

        # Initialize lists to store decoder outputs
        trg_tokens = []

        # Iterate until maximum length or until <eos> token is generated
        for _ in range(MAX_LENGTH):
            # Pass input, previous hidden state, and encoder outputs to decoder
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)

            # Get the predicted token
            pred_token = output.argmax(1).item()

            # Append predicted token to list
            trg_tokens.append(pred_token)

            # Break if <eos> token is generated
            if pred_token == s_idx_to_char['<eos>']:
                break

            # Update decoder input with predicted token for next iteration
            trg_tensor = torch.tensor([[pred_token]], device=device)

        # Convert token indices to characters
        translated_sentence = [t_idx_to_char[token] for token in trg_tokens]

        return translated_sentence

# Example usage
input_sentence = [1, 2, 3, ...]  # Replace with your input sentence
output_sentence = inference(input_sentence)
print(output_sentence)
