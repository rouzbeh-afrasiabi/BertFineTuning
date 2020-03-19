from BertFineTuning.data_config import *

from pytorch_transformers import BertTokenizer

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class CustomSet(Dataset):
    def __init__(self,_target,_max_token_length):
        self.samples = pd.read_csv(_target, index_col=[0])
        self.max_length=_max_token_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text=str(self.samples['text'].iloc[idx])
        label_names=[col  for col in self.samples.columns if 'label' in col]
        id_names=[col  for col in self.samples.columns if 'id' in col]
        if(id_names):
            ids=self.samples[id_names[0]].iloc[idx]
        else:
            ids=[]
        labels=tuple(self.samples[label].iloc[idx] for label in label_names)
        tokenized_text = np.array(tokenizer.tokenize(text))
        list_of_indices=torch.tensor(tokenizer.convert_tokens_to_ids(tokenized_text)).to(device)
        segments_ids=torch.tensor([int(102 in list_of_indices[:i]) for i,index in enumerate(list_of_indices)]).to(device)
        new_list_of_indices=list_of_indices[0:self.max_length-1].to(device)
        new_segments_ids=segments_ids[0:self.max_length-1].to(device)
        pad_len=(self.max_length-len(new_list_of_indices))
        if(pad_len>0):
            new_list_of_indices=F.pad(new_list_of_indices, pad=(0,pad_len), mode='constant', value=0).to(device)
            new_segments_ids=F.pad(new_segments_ids, pad=(0,pad_len), mode='constant', value=0).to(device)
        return (ids,new_list_of_indices,new_segments_ids)+labels

    
    
    
class MultiLoader():
    def __init__(self,):
        self.__kwargs=data_load_config()
        self.__keys=list(self.__kwargs.keys())
        self.__values=list(self.__kwargs.values())
        self.__items=self.__kwargs.items()
        if(not any([key in self.__dict__ for key in self.__kwargs.keys()])):
            self.__dict__.update(self.__kwargs)
            
        def DataLoaders():
            output_sets={}
            output_loaders={}
            for key,location in self._DataLocation.items:
                output_sets[key]=CustomSet(location,self._max_token_length)
                if hasattr(self._DataLoader_config, key):
                    output_loaders[key]=DataLoader(output_sets[key], **getattr(self._DataLoader_config,key))
                else:
                    warnings.warn(key+" is not a valid key.")
            return(output_loaders)
        self.__dict__.update(DataLoaders())
            
                