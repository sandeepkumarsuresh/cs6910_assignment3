from DataPreprocess import *
from Model import *



if __name__ == '__main__':




    # # adjustable parameters
    # INPUT_DIM = len(source.vocab)
    # OUTPUT_DIM = len(target.vocab)
    # ENC_EMB_DIM = 256
    # DEC_EMB_DIM = 256
    # HID_DIM = 512
    # N_LAYERS = 2
    # ENC_DROPOUT = 0.5
    # DEC_DROPOUT = 0.5



    path_to_dataset = 'Dataset/mal/mal_train.csv'

    x = LoadDataset(path_to_dataset)
    a,b = x.load()

    y = Tokenizer()

    c = y.tokenizer(a)
    print(c)
    d,e = y.token_mapping(c)

    # print(d)
    # print(e)

    # f = y.convert_tensor_word(word='hello',num_tokens=d,token_index_map=e)

    f = y.tensor_mapping(char='h',num_tokens=d,token_index_map=e)


    print(f.dtype,'shape',f.shape)
    
    n_hidden = 128
    enc_config = Encoder_Config(hidden_size=256,n_layers=2)
    rnn = EncoderRNN(enc_config)
    o,h = rnn(f)

    print(o.shape)
    print(h.shape)

    # de_config = Decoder_Config(input_size=)

    # input = letterToTensor('A')
    # hidden = torch.zeros(1, n_hidden)

    # output, next_hidden = rnn(input, hidden)