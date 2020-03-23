import os
import BertFineTuning.utils


cwd = os.getcwd()
processed_data_folder=os.path.join(cwd,'data','processed')
target_folder=os.path.join(cwd,'data','processed')


max_string_length=150 


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
}


# class DataLocation():
#     def __init__(self,):
        
#         def get_loc(target_folder=processed_data_folder):
#             files=os.listdir(target_folder)
#             result={os.path.splitext(file_name)[0]:os.path.join(target_folder,file_name) for file_name in files}
#             return result
#         kwargs=get_loc()
#         self.__kwargs=kwargs
#         self.keys=list(kwargs.keys())
#         self.values=list(kwargs.values())
#         self.items=kwargs.items()
#         if(not any([key in self.__dict__ for key in self.__kwargs.keys()])):
#             self.__dict__.update(kwargs)

# class LoaderConfig():
#     def __init__(self,):
#         self.__kwargs=DataLoader_config
#         self.__loc=DataLocation()
#         if(not any([key in self.__dict__ for key in self.__kwargs.keys()])):
#             self.__dict__.update(dict(filter(lambda target: target[0] in self.__loc.keys, self.__kwargs.items())))
            

# def data_load_config():
#     return {'_DataLocation':DataLocation(),'_DataLoader_config':LoaderConfig(),'_max_token_length':max_token_length}

        