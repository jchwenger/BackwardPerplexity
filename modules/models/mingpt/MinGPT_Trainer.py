import wandb
import random

import torch
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from torchenhanced import Trainer

from .MinGPT import MinGPT
from ...datasets import TokenText
from ...tokenizer import Tokenizer


class MinGPT_Trainer(Trainer):
    def __init__(self, model: MinGPT, train_dataset: TokenText, valid_dataset : TokenText, backwards : bool=True, 
                 detokenizer :Tokenizer=None, optim: Optimizer = None, scheduler: _LRScheduler = None, 
                 state_save_loc=None, device: str = 'cpu',parallel=None, run_name: str = None, project_name: str = None,
                 run_config: dict ={}):
        super().__init__(model, optim, scheduler, save_loc=state_save_loc, device=device, parallel=parallel,
                         run_name=run_name, project_name=project_name, run_config=run_config)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        # For logging :
        self.batch_loss = []

        self.detokenizer= detokenizer

        self.backwards = backwards # In principle, extractable form dataset, but annoying because I use Subset, so I just give it.

        # Print number of parameters
        print(f"Number of parameters : {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")

    def get_loaders(self,batch_size,num_workers=0):
        # note shuffle= False because both are pre-shuffled; that way we can restart deterministically from same point
        t_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        v_dataloader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.text_table = wandb.Table(columns=["batches","text"])

        return t_dataloader, v_dataloader

    def process_batch(self, batch_data):
        # Compute the loss, and log the learning rate
        loss = self.compute_loss(batch_data)
        
        if(self.do_step_log) :
            self.logger.log({'lr' : self.scheduler.get_last_lr()[0]},commit=False)
    
        return loss

    def compute_loss(self, batch_data):
        token_in, token_truth = batch_data
        B,T = token_truth.shape

        token_in = token_in.to(self.device)
        token_truth = token_truth.to(self.device) # (B, T)

        token_in = self.model.forward(token_in)
        # RESHAPE OTHERWISE, CROSS ENTROPY LOSS BUGS FOR BIG BATCHES
        token_in = token_in.reshape(B*T,-1) # (B*T, C)
        token_truth = token_truth.reshape(B*T)

        loss = F.cross_entropy(token_in,token_truth,reduction='mean')

        return loss

    def process_batch_valid(self, batch_data):
        loss = self.compute_loss(batch_data)

        return loss

    def valid_log(self):
        """
            Log a snippet of generated text in a wandb Table
        """
        data, _ = self.valid_dataset[random.randint(0,len(self.valid_dataset)-1)] # (T,)*2
        data = data[:5].to(self.device) # only keep first 5 tokens, to start generating
        if(self.parallel_train):
            modello = self.model.module
        else:
            modello = self.model
        # Generate some tokens
        phrase_out = modello.generate(data[None,:],max_new_tokens=100, do_sample=True).cpu() # (1, 5+300)

        # Decode the tokens, and if backwards, flip them back
        if(self.backwards):
            phrase_out= self.detokenizer.detokenize(torch.flip(phrase_out,dims=[1])) # ()
        else :
            phrase_out=self.detokenizer.detokenize(phrase_out)

        self.text_table.add_data(f"{self.steps_done/1000:.1f}k",phrase_out) 
        # Fucking wandb... To update the table before end, need to re-create one each time
        new_table = wandb.Table(
        columns=self.text_table.columns, data=self.text_table.data
        )
        
        wandb.log({'gen_samples': new_table},commit=False)
