<h2>CS6910 Assignment 3 - Transliteration </h2>


## Prerequisites

Before running the code you need to install all the requirements

```
pip3 install -r requirements.txt
```
The below code is set to default Vannila RNN Configurations

**To Run the Code**

* Without Attention 

```
usage: python3 main_vannilla.py 
               [--epochs EPOCHS] 
               [--optimizer OPTIMIZER]
               [--cell_type CELL_TYPE] 
               [--learning_rate LEARNING_RATE]
               [--batch_size BATCH_SIZE] 
               [--embedding_size EMBEDDING_SIZE]
               [--hidden_size HIDDEN_LAYER_SIZE]
               [--dropout DROPOUT]
               [--bidirectional BIDIRECTIONAL]
               [--teacher_forcing TEACHER_FORCING]
```

* With Attention

```
usage: python attention_main.py 
               [--epochs EPOCHS] 
               [--optimizer OPTIMIZER]
               [--cell_type CELL_TYPE] 
               [--learning_rate LEARNING_RATE]
               [--batch_size BATCH_SIZE] 
               [--embedding_size EMBEDDING_SIZE]
               [--hidden_size HIDDEN_LAYER_SIZE]
               [--dropout DROPOUT]
               [--bidirectional BIDIRECTIONAL]
               [--teacher_forcing TEACHER_FORCING]

```

**Files**

* main_vannilla.py -- main funtion for running vanilla RNN
* main_attn.py -- main funtion (attention)
* Config.py -- to set up the configuration files
* Datapreprocess.py -- To build the vocabulary
* Load_batch.py -- for batching the data
* Model.py -- for creating vannilla model architecture
* Model_Attention.py -- for creating attenton model architecture
* prediction_wandb.py -- to do wandb run of the predictions
* train.py -- for training the Model
* Transliteration_Dataloader.py -- for setting up the Transliteration Dataloader
* Utility.py -- Contains all the helper functions

## Author

* **Sandeep Kumar Suresh** - *Github Link* - [Github](https://github.com/sandeepkumarsuresh/cs6910_assignment3)
                            *Wandb Link* - [Wandb](https://api.wandb.ai/links/ee23s059/zg89vvb7)