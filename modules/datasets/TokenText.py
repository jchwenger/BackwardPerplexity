from torch.utils.data import Dataset
import torch,os, h5py
import numpy as np


class TokenTextBOS(Dataset):
    """
        Dataset used to store tokenized text. Produces tuples of tokenized text, 
        to be used as input and target for language modelling. Uses memory mapping, with hdf5.
        Adds a BOS token at the beginning, such that we don't 'miss' any token. As such, the 'effective'
        attention length is one less, as one token is dedicated to the <BOS> token.

        Args:
        text_location : location of the tokenized text tensor
        attn_length : size of the attention window (including the BOS token)
        stride : by how many tokens to stride to get the next example. Default is half the attention length.
        backwards : Whether to train with text backwards.
    """

    def __init__(self,h5_file :str, attn_length:int, stride:int=None, backwards=False):
        self.h5_file = h5_file
        self.attn_length = attn_length-1 # -1 because we add a BOS token
        self.backwards = backwards

        
        if(stride is None):
            self.stride=self.attn_length//2
        else :
            self.stride = stride

        if(not os.path.isfile(self.h5_file)):
            raise ValueError(f'File/Folder {self.h5_file} not found')
        
        self.h5_file = h5py.File(self.h5_file, 'r')
        self.text_tensor = self.h5_file['tokens']


        self.num_tokens = len(self.text_tensor)
        self.length = (self.num_tokens-self.attn_length-1)//(self.stride) # -1 because we need to have a target for each input

        print(f'Dataset contains {self.num_tokens/1e6:.2f}M tokens, resulting in {self.length//1000}k examples.')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
            Returns a tuple of (input, target) tensors, each of shape (attn_length)
            If self.backwards = True, the examples are flipped around.
        """
        true_idx = self.stride*(idx)

        if(self.backwards):
            return self.add_BOS(torch.tensor(self.text_tensor[true_idx+1:true_idx+self.attn_length+1],dtype=torch.long).flip(dims=(0,))), \
                torch.tensor(self.text_tensor[true_idx:true_idx+self.attn_length+1],dtype=torch.long).flip(dims=(0,))
        else :
            return self.add_BOS(torch.tensor(self.text_tensor[true_idx:true_idx+self.attn_length],dtype=torch.long)), \
            torch.tensor(self.text_tensor[true_idx:true_idx+self.attn_length+1],dtype=torch.long)

    def add_BOS(self,tens):
        """
            Adds a BOS token at the beginning of the tensor, and returns it.

            Args:
            tens : tensor of shape (attn_length)
        """
        return torch.cat([torch.tensor([0],dtype=torch.long),tens],dim=0) # (attn_length+1)
    