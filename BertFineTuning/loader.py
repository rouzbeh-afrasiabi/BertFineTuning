
from BertFineTuning.loader_config import default_loader_config
from transformers import BertTokenizer


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import warnings
import logging
import multiprocessing as mp

mp.set_start_method('spawn')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def data_load_config(target_folder,dataloader_config,max_token_length):
    return {'_DataLocation':DataLocation(target_folder),'_DataLoader_config':LoaderConfig(target_folder,dataloader_config),
            '_max_token_length':max_token_length}

class DataLocation():
    def __init__(self,target_folder):
        self.target_folder=target_folder
        
        def get_loc():
            files=os.listdir(self.target_folder)
            result={os.path.splitext(file_name)[0]:os.path.join(target_folder,file_name) for file_name in files}
            return result
        kwargs=get_loc()
        self.__kwargs=kwargs
        self.keys=list(kwargs.keys())
        self.values=list(kwargs.values())
        self.items=kwargs.items()
        if(not any([key in self.__dict__ for key in self.__kwargs.keys()])):
            self.__dict__.update(kwargs)

class LoaderConfig():
    def __init__(self,target_folder,dataloader_config):
        self.__kwargs=dataloader_config
        self.__loc=DataLocation(target_folder)
        if(not any([key in self.__dict__ for key in self.__kwargs.keys()])):
            self.__dict__.update(dict(filter(lambda target: target[0] in self.__loc.keys, self.__kwargs.items())))
            


class CustomSet(Dataset):
    def __init__(self,_target,_max_token_length):
        
        
        self.samples = pd.read_csv(_target, index_col=[0],engine='python')
        self.max_length=_max_token_length
        self.samples=self.samples.loc[self.samples.total_tokens<=_max_token_length]


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
        labels=list(self.samples[label].iloc[idx] for label in label_names)
        list_of_indices = tokenizer.encode(text)
        segments_ids=torch.tensor([int(102 in list_of_indices[:i]) for i,index in enumerate(list_of_indices)])
        
        new_list_of_indices=torch.tensor(list_of_indices[0:self.max_length-1])
        new_segments_ids=segments_ids[0:self.max_length-1]
        pad_len=(self.max_length-len(new_list_of_indices))
        if(pad_len>0):
            new_list_of_indices=F.pad(new_list_of_indices, pad=(0,pad_len), mode='constant', value=0).to(device)
            new_segments_ids=F.pad(new_segments_ids, pad=(0,pad_len), mode='constant', value=0).to(device)
        return {'ids':ids,'list_of_indices':new_list_of_indices,'segments_ids':new_segments_ids,'labels':labels}

#to iclude later

# def collate_fn(data):
#     """
#        data: is a list of tuples with (example, label, length)
#              where 'example' is a tensor of arbitrary shape
#              and label/length are scalars
#     """
#     ids,list_of_indices,segments_ids,labels=data.values()
#     print(data)
#     max_len = max(lengths)
#     n_ftrs = data[0][0].size(1)
#     features = torch.zeros((len(data), max_len, n_ftrs))
#     labels = torch.tensor(labels)
#     lengths = torch.tensor(lengths)

#     for i in range(len(data)):
#         j, k = data[i][0].size(0), data[i][0].size(1)
#         features[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])

#     return features.float(), labels.long(), lengths.long()    
    
    
class MultiLoader():
    def __init__(self,target_folder,dataLoader_config,model_config,show_warning=True):
        self.show_warning=show_warning
        self.__kwargs=data_load_config(target_folder,dataLoader_config,model_config['max_token_length'])
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
                    output_loaders[key]=DataLoader(output_sets[key],**default_loader_config)
                    if(self.show_warning):
                        logging.warn(key+".csv"+"\nFile does not have a valid  data loader configuration in dataloader_config"
                        +", using default_loader_config.")
                output_loaders[key].max_token_length=self._max_token_length
            return(output_loaders)
        self.__dict__.update(DataLoaders())
            
                