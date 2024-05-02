"""
    Training script for GPT-like models. Use this along with the output of gen_run !

    Usage : python train_script.py <path_to_json_config_file> -d <device> -t <tokenizer_path> -p <project_name> -s

    Example : python train_script.py TrainParams/params.json -d cuda:0 -t fr -p BackPerplexityResilient -s

    Note : uses wandb, so you need to have a wandb account and be logged in.
"""
from modules import *
import torch, torch.optim,os, argparse,json, pathlib,random, shutil
from torch.utils.data import Subset
from torchenhanced import CosineWarmup
import numpy as np


def train(file_location : str, device : str, tokenizer_path : str ,
           project_name :str =None, step_pickup: bool=True):
    """
        Main train script. Launches training of a model given parameters in a JSON file at file_location

        Args:
        file_location : path to the JSON config file
        device : device to train on. 'cpu' or 'cuda:0' etc.
        tokenizer_path : path to the tokenizer to use. Relative to the train_script folder.
        project_name : name of the project to log to. Default is 'NewBackPerp'
        step_pickup : If false, train steps_to_train steps more. If true, will train UP TO steps_to_train TOTAL steps.
    """
    if(project_name is None):
        project_name = 'NewBackPerp'
    
    cur_path = pathlib.Path(__file__).parent.absolute().as_posix()
    run_name = os.path.splitext(os.path.basename(file_location))[0]

    print('TOKENIZER PATH : ', os.path.join(cur_path,tokenizer_path))
    if(tokenizer_path is None):
        raise(ValueError('Tokenizer path required, please specify with -t <tokenizer_path>'))
    if(not os.path.exists(os.path.join(cur_path,tokenizer_path))):
        raise(FileNotFoundError(f'Tokenizer not found at path {os.path.join(cur_path,tokenizer_path)}. \n \
                                Tokenizer path should be relative to train_script.py.'))
    
    tokenizer_path = os.path.join(cur_path,args.tokenizer_path)
    tokenizer = get_tokenizer(m_path=tokenizer_path)

    try:
        with open(file_location,'r') as f :
            configo = json.load(f)
            model_params = configo['model_params']
            training_params = configo['training_params']
            optim_params = configo['optim_params']
    except Exception as e:
        print(f'Error reading JSON file !')
        raise(e)

    if not os.path.exists(training_params['dataset_folder']):
        raise FileNotFoundError(f"Tried to find dataset folder at \
                                {training_params['dataset_folder']}, but failed. \
                                Make sure there is the folder {training_params['dataset_folder']}\
                                in the same directory.")


    valid_steps =training_params['valid_steps'] # We ask how many validation steps. To get these, we assume 5% of training time allocated for validation.
    valid_percent_time=5 # Time spent validating, in percentage
    valid_percent_time=valid_percent_time/100

    valid_every = int(valid_steps/valid_percent_time)
    
    backwards = training_params['backwards'] # If we train backwards

    rng = np.random.default_rng(42) # For deterministic shuffling of dataset

    # First copy the dataset in the current folder. 
    # This is useful in the case of a network drive, where the dataset is slow to access.
    # Can be removed if dataset location is local.

    dataset_path = training_params['dataset_folder']
    destination_path = os.path.join(cur_path,'local_dataset.h5')

    if(not os.path.exists(destination_path)):
        print('Copying dataset to current folder...')
        shutil.copy(dataset_path, destination_path)
    else :
        print('Dataset already copied to current folder, using this one.')
    
    # Generate and shuffle the dataset
    motherDataset = TokenTextBOS(h5_file=destination_path, attn_length=model_params['attn_length'], backwards=backwards)
    indices = np.arange(len(motherDataset))
    rng.shuffle(indices)
    motherDataset = Subset(motherDataset, list(indices)) # Shuffled dataset

    # To keep it constant even if we switch batch_size, I take 250*valid_steps datapoints in the validation set
    val_inds = valid_steps*250
    val_range = range(len(motherDataset)-val_inds,len(motherDataset)) # Validation, last portion of dataset
    keep_range = range(len(motherDataset)-val_inds) # Training, first portion of dataset
    
    del indices

    #Whether backwards or forwards, its the individual examples that are flipped, not the dataset. So same thing for both !
    train_dataset = Subset(motherDataset, keep_range)
    val_dataset = Subset(motherDataset, val_range)

    print('Datapoint example. Check it looks correct ! : ')
    print(f'TRAIN DATA   :',tokenizer.detokenize(train_dataset[0][0][:20]))
    print(f'TRAIN ANSWER : ',tokenizer.detokenize(train_dataset[0][1][:20]))
    print(f'VALID DATA   :',tokenizer.detokenize(val_dataset[0][0][:20]))
    print(f'VALID ANSWER : ',tokenizer.detokenize(val_dataset[0][1][:20]))

    model = MinGPT(**model_params)
    #====================== TRAINING PARAMETERS =======================
    batch_size = training_params['batch_size']
    aggregate = training_params['aggregate']
    totbatches = len(train_dataset)//batch_size
    
    if(training_params['steps_to_train']==None):
        steps_to_train = totbatches
    else:
        steps_to_train = training_params['steps_to_train']

    print(f'{totbatches=}, {batch_size=}, {len(train_dataset)=}')
    base_lr = optim_params['lr']
    warmup_steps = optim_params['warmup_steps']
    oscil_steps = optim_params['oscil_steps'] # Period of cosine restart for oscillation
    lr_shrink_factor=optim_params['lr_shrink']
    lr_min = optim_params['lr_min']
    lr_init = optim_params['lr_init']

    print(f'--- Training for ~ {steps_to_train//1000}k minibatches ---')
    #------ Optimizers ------
    optim = torch.optim.AdamW(model.parameters(), lr=base_lr)
    # Cosine schedule with Warmup, from torchenhanced package
    scheduler = CosineWarmup(optim,warmup_steps=warmup_steps,lr_shrink=lr_shrink_factor,
                             lr_init=lr_init,T_0=oscil_steps,eta_min=lr_min)

    # Intialize the trainer class
    trainer = MinGPT_Trainer(
        model=model,optim=optim,scheduler=scheduler,
        train_dataset=train_dataset,valid_dataset=val_dataset, detokenizer=tokenizer,
        run_name=run_name, project_name=project_name, state_save_loc=training_params['state_save_loc'], backwards=backwards,
        device=device, run_config={'model_params':model_params,'train':training_params,'opti':optim_params} 
        )

    
    if(os.path.exists(os.path.join(training_params['state_save_loc'],project_name,'state',run_name+'.state'))):
        trainer.load_state(os.path.join(training_params['state_save_loc'],project_name,'state',run_name+'.state'))
    trainer.stepnum =1
    trainer.train_steps(steps=steps_to_train,save_every=2000,aggregate=aggregate,
                        backup_every=training_params['backup_every'],step_log=training_params['step_log'],
                        batch_size=batch_size,valid_every=valid_every,resume_batches=True,pickup=step_pickup)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Starts training of Predictor model given a JSON config file.")
    parser.add_argument("file_location", help="Path to the JSON config file. Relative to where you launch the script from.")
    parser.add_argument("-d", "--device", type=str, default='cpu', help="Device string, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("-t", "--tokenizer_path", type=str,help="Path for the tokenizer to use (only used for logging snippets). Relative to the train_script folder.")
    parser.add_argument("-p", "--project_name", help="Name of the project to log to. Default is 'BackPerplexityResilient'")
    parser.add_argument("-s", "--no_step_pickup", action="store_false", help="If set, train steps_to_train steps more. Otherwise, will train UP TO steps_to_train TOTAL steps.")  
    args = parser.parse_args()

    train(args.file_location, device=args.device, tokenizer_path=args.tokenizer_path, 
          project_name=args.project_name, step_pickup=args.no_step_pickup)