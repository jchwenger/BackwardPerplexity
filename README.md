# BackPerplexity
Investigate perplexity of LLM's when trained backward vs forward


Code used for the Natural Language experiments of the paper 'Arrows of Time in Large Langugage Models'.

Install requirements using `pip install -r requirements.txt`.
NOTE : On Windows, doing this might install torch without CUDA support. If this is the case, first install pytorch CUDA following instruction on the official [website](https://pytorch.org/), then run `pip install -r requirements.txt`.

Read the following section to learn how to reproduce experiments.

## Tokenization
The script `tokenize_to_h5.py` can be used to prepare a dataset for training. Given a .txt file, it will train a BPE tokenizer on it, then use it to tokenize the text, and save the tokenized dataset in `.h5` format.

CC100 datasets can be downloaded [here](https://data.statmt.org/cc-100/). 
### Usage :
To use `tokenize_to_h5.py`, first put a standalone `.txt` file inside a folder. Then, inside `tokenize_to_h5.py`, modify the following :
``` 
if __name__=='__main__':
    txt_path = '' # Path to the .txt file to be tokenized
    out_h5_folder = '' #  Folder that will contain the output .h5 file
    tokenizer_folder = '' # Folder where the tokenizer will be saved
    tokenizer_name = '' # Name of the tokenizer that will be saved
```

Then run the script. NOTE : tokenization of large .txt files (>100GB) might take a while (1,2 days). This script is NOT designed to pick up where it left off if it crashes. For bigger datasets, consider making a script (include `from modules.tok_utils import *`), and run, subsequently :
    - `create_tokenizer(txt_path, tokenizer_folder,tokenizer_name=tokenizer_name)` : Will train the BPE tokenizer on the given .txt file, and save it in <tokenizer_folder>/<tokenizer_name>
    - `tokenize_folder(os.path.dirname(txt_path), os.path.join(tokenizer_folder,tokenizer_name))` : Will tokenize the text file, splitting it into subfiles if necessary for memory reasons. Saved the tokenized tensors as `.pt`. If it crashes mid-way, can be restarted, and will pickup from last checkpoint
    - `make_h5(os.path.dirname(txt_path), out_h5_folder,toki)` : Will convert a folder containing `.pt` files into a single `.h5` dataset, ready for training.

For more informations on these functions, look at docstring comments in `modules/tok_utils`

### Tokenizer class
The tokenizer class we use throughout the project is defined in `modules/tokenizer.py`. It is a wrapper on top of the Huggingface tokenizer.

Here is all you need to know to use the tokenizers :

```
from modules import tokenizer

toki = tokenizer.get_tokenizer(m_path='modules/tokenizers/en_tokenizer') # Load a saved tokenizer by specifying saved folder
# A saved tokenizer is created by using create_tokenizer in modules/tok_utils/create_custom_tokenizer.py

tokenized = toki.tokenize("Hello, world!") # Tokenize a string
print(tokenized) # Get a tensor of ints shape [1, seq_len]
print(toki.detokenize(tokenized)) # Detokenize a tensor of ints, prints "Hello, world!"
```

## Training

### Scripts
For training, 4 scripts are provided. All are designed to train models on the dataset generated with the above method.
- `train_gru.py` : Trains GRU model. (CURRENTLY UPDATING TO NEW VERSION OF TORCHENHANCED)
- `train_lstm.py` : Trains LSTM model.(CURRENTLY UPDATING TO NEW VERSION OF TORCHENHANCED)
- `train_parallel.py` : Trains GPT model on multiple GPUs, using `torch.nn.Dataparallel` (CURRENTLY UPDATING TO NEW VERSION OF TORCHENHANCED)
- `train_script.py` : Trains GPT model on a single GPU. Up to date


For all 4 scripts, usage is as follows :
```usage: train_script.py [-h] [-d DEVICE] [-t TOKENIZER_PATH] [-p PROJECT_NAME] [-s] file_location

Starts training of Predictor model given a JSON config file.

positional arguments:
  file_location         Path to the JSON config file. Relative to where you launch the script from.

options:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                Device string, e.g. 'cuda:0' or 'cpu'
  -t TOKENIZER_PATH, --tokenizer_path TOKENIZER_PATH
                Path for the tokenizer to use (only used for logging snippets). Relative to the train_script folder.
  -p PROJECT_NAME, --project_name PROJECT_NAME
                Name of the project to log to. Default is 'BackPerplexityResilient'
  -s, --no_step_pickup 
                If set, train steps_to_train steps more. Otherwise, will train UP TO steps_to_train TOTAL steps."
  ```

Example :
`python train_script.py path/to/config.json -d cuda:0 -t path/to/tokenizer -p MyTrainingProject -s`
### JSON config file
To run the training script, we need to provide it with a path to the JSON config file. Their format slightly depends if training a GPT, GRU or LSTM model. In a nutshell, they contain all the necessary hyperparameters for a training run.

Here is a description of each entry : 

```
{
    "model_params": { # Model parameters
        "vocab_size": 50257, # Vocabulary size
        "n_layers": 12, # Number of Transformer Blocks
        "n_heads": 12, # Number of attention heads
        "embed_dim": 768, # Number of hidden/embedding dimensions
        "attn_length": 256, # Attention Length
        "mlp_ratio": 4.0, # MLP ratio
        "dropout": 0.1, # Dropout inside tranformer blocks
        "embd_dropout": null # Dropout for the token embeddings. Defaults to 0.
    },
    "training_params": { 
        "dataset_folder": "english/english.h5", # Location of .h5 dataset to train on
        "batch_size": 180, # Batch size
        "aggregate": 1, # Number of times to aggregate gradients before gradient step. (effective batch_size = aggregate*batch_size)
        "backwards": false, # Whether to train in the backwards direction
        "steps_to_train": null, # Number of gradient steps to train. Defaults to one epoch of the dataset.
        "save_every": 3000, # Number of steps between each save of the training state.
        "backup_every": 15000, # Number of steps between a backup of the training state.
        "step_log": 400, # Number of steps between each log of training loss in wandb
        "valid_steps": 1000, # Number of batches seen during one validation.
        "state_save_loc": "datavol/vassilis/runs" # folder in which to save the training state.
    },
    "optim_params": {
        "lr": 0.0001, # Base learning rate
        "warmup_steps": 4000, # Number of batches until learning rate warms up
        "oscil_steps": 300000, # Number of steps between warm restarts
        "lr_shrink": 0.85, # Shrinking factor of lr between warm restarts
        "lr_init": 1e-07, # Initial learning rate, for warmup
        "lr_min": 1e-06 # Minimum learning rate reached during cosine annealing.
    }
}
```