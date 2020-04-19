import os
import BertFineTuning.utils
import multiprocessing as mp

cwd = os.getcwd()
processed_data_folder=os.path.join(cwd,'data','processed')
target_folder=os.path.join(cwd,'data','processed')
mp.set_start_method('spawn')

max_string_length=150 
max_token_length=100

DataLoader_config_default={
 'batch_size': 20,
 'shuffle': True,
 'sampler': None,
 'batch_sampler': None,
 'num_workers': 0,
 'pin_memory': False,
 'drop_last': False,
 'timeout': 0,}

# 'collate_fn': None,
#  'worker_init_fn': None,
#  'multiprocessing_context': None}


DataLoader_config={
'bert_train_split':DataLoader_config_default,
'bert_test_split':DataLoader_config_default,
'bert_valid_split':DataLoader_config_default,
'test':1
}

        