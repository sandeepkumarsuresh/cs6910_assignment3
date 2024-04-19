from DataPreprocess import *
from Model import *



if __name__ == '__main__':

    path_to_dataset = 'Dataset/mal/mal_train.csv'

    x = LoadDataset(path_to_dataset)
    a,b = x.load()

    y = Tokenizer()

    c = y.tokenizer(a)

    d,e = y.token_mapping(c)

    # print(d)
    # print(e)

    # f = y.convert_tensor_word(word='hello',num_tokens=d,token_index_map=e)

    f = y.tensor_mapping(char='h',num_tokens=d,token_index_map=e)


    print(f.dtype,'shape',f.shape)
    
    n_hidden = 128

    rnn = EncoderRNN(hidden_size=n_hidden)
    o,h = rnn(f)

    print(o.shape)
    print(h.shape)


    # input = letterToTensor('A')
    # hidden = torch.zeros(1, n_hidden)

    # output, next_hidden = rnn(input, hidden)