import pandas as pd
class LoadDataset:
    def __init__(self,dataset_path):
        self.dataset_path = dataset_path
        self.input = []
        self.output = []

    
    def load(self):
        """
        This function loads the dataset and returns it as a list

        """

        df = pd.read_csv(self.dataset_path)
        
        self.input = df.iloc[:,0].to_list()   
        self.output = df.iloc[:,1].to_list()

        return self.input ,self.output

# if __name__ == '__main__':

#     path_to_dataset = 'Dataset/mal/mal_train.csv'

#     x = LoadDataset(path_to_dataset)
#     a,b = x.load()
