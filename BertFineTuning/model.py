from BertFineTuning.utils import *
from BertFineTuning.model_config import*

import os
import sys
import numpy as np
import pandas as pd
from collections import OrderedDict
from pycm import *

import torch
if(torch.cuda.is_available()):
    torch.cuda.current_device()
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import copy
import gc

from pytorch_transformers import BertModel

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.insert(0, cwd)




random_state=123
torch.manual_seed(random_state)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)




class BertFineTuning():
    def __init__(self,):
        
        class Network(nn.Module):
            def __init__(self, pre_trained_model,config):
                super().__init__()
                self.pre_trained_model=pre_trained_model.to(config['device'])
                self.classifier=nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(config['in_features'], config['in_features'])),
                    ('bn_1',nn.BatchNorm1d(config['in_features'])),
                    ('prelu1', nn.PReLU()),
                    ('fc2', nn.Linear(config['in_features'], config['num_classes']))]))

            def forward(self, tokens_tensor, segments_tensors):
                last_hidden_state, pooled_output  = self.pre_trained_model(tokens_tensor, segments_tensors)
                logits = self.classifier(pooled_output)
                return logits 
            
        def __device():
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.check_point_loaded=False
        self.device = __device()
        self.criterion_config={}
        self.optimizer_config={}
        self.scheduler_config={}
        self.config=model_config
        self.config['device']=self.device
        self.pre_trained_model=BertModel.from_pretrained('bert-base-uncased')
        self.model=Network(self.pre_trained_model,self.config).to(self.device)
        self.parameters_main=[
            {"params": self.model.pre_trained_model.parameters(),
             "lr": self.config['learning_rate_PT'],'weight_decay': self.config['weight_decay']},
            {"params": self.model.classifier.parameters(),
             "lr": self.config['learning_rate_CLS'],'weight_decay': self.config['weight_decay']},
                                ]
        self.no_decay = ['bias', 'LayerNorm.weight']
        self.__PT_n_param=self.model.pre_trained_model.named_parameters()
        self.__CLS_n_param=self.model.classifier.named_parameters()
        self.parameters_noDecay=[
            {'params': [p for n, p in self.__PT_n_param if not any(nd in n for nd in self.no_decay) and p.requires_grad],
             "lr": self.config['learning_rate_PT'], 'weight_decay': self.config['weight_decay']},
            {'params': [p for n, p in self.__PT_n_param if any(nd in n for nd in self.no_decay) and p.requires_grad],
             "lr": self.config['learning_rate_PT'], 'weight_decay': 0.0},
            {'params': [p for n, p in self.__CLS_n_param if p.requires_grad],
             "lr": self.config['learning_rate_PT'], 'weight_decay': self.config['weight_decay']},
                                ]
        self.criterion=None
        self.optimizer=None
        self.scheduler=None
        self.validate_at_epoch=0
        self.checkpoint=None
        self.loss_history=[]
        self.test_loss_history=[]
        self.learning_rate=[]
        self.cm_test=[]
        self.cm_train=[]
        self.last_epoch=0
        self.epochs=100
        self.validate_at_epoch=0
        self.print_every=100
        self.e=0
        self.target_folder=cwd
        self.save_folder=os.path.join(cwd,'checkpoints')
        
    @staticmethod
    def _update_dict_strict(target,**kwargs):
        if(all([key in target.keys() for key in kwargs.keys()])):
            target.update(kwargs)
        else:
            raise Exception('Following keys not in dictionary',[key for key in kwargs.keys() if(key not in target.keys())])  
    
    @staticmethod       
    def _update_dict(target,**kwargs):
        target.update(kwargs) 
        
    def update_config(self,**kwargs):
        self._update_dict_strict(self.config,**kwargs) 
        
    @staticmethod
    def print_results(cm):
        print(cm.AUCI)
        print("MCC: ",cm.MCC)
        print("Accuracy: ",cm.ACC)
        print({"F1 Macro ":cm.F1_Macro},{'F1 Micro':cm.F1_Micro})
        print({"F1 ":cm.F1})
        print("Precision: ",cm.PPV)
        print("recall: ",cm.TPR)
        cm.print_matrix() 

    def save_it(self,target_folder):
        self.model.eval()
        print("Saving Model ...")
        checkpoint = {'state_dict': self.model.state_dict(),
                    'optimizer':self.optimizer.state_dict(),
                    'optimizer_type':type(self.optimizer),
                    'criterion':self.criterion,
                    'criterion_type':type(self.criterion),
                    'scheduler':self.scheduler.state_dict(),
                    'scheduler_type':type(self.scheduler),
                    'last_epoch':self.e+1,
                    'train_loss_history':self.loss_history,
                    'test_loss_history':self.test_loss_history,
                    'learning_rate_history':self.learning_rate,
                    'cm_train':self.cm_train,
                    'cm_test':self.cm_test,
                    'config':self.config,
                    'train_loops':self.train_loops
                  }
        try:
            torch.save(checkpoint,target_folder+'/'+'checkpoint'+str(self.e+1)+'.pth' )
            print("Model Saved.\n")
            self.model.train()
        except:
            print("Failed to Save Model!!")
            
    def load_checkpoint(self,path):
        if(check_file(path)):
            self.checkpoint = torch.load(path,map_location=self.device)
            self.model.load_state_dict(self.checkpoint["state_dict"])
            self.optimizer.load_state_dict(self.checkpoint["optimizer"])
            self.scheduler.load_state_dict(self.checkpoint["scheduler"])
            
            self.loss_history=self.checkpoint['train_loss_history']
            self.test_loss_history=self.checkpoint['test_loss_history']
            self.learning_rate=self.checkpoint['learning_rate_history']
            self.cm_test=self.checkpoint['cm_test']
            self.cm_train=self.checkpoint['cm_train']
            self.last_epoch=self.checkpoint['last_epoch']
            self.check_point_loaded=True
            self.model.eval()
            return 

    def predict(self,target_loader):
        self.model.eval()
        with torch.no_grad():
            criterion=self.criterion
            predictions=np.array([])
            loss_history=[0]
            labels=np.array([])
            for i, (_list_of_indices,_segments_ids,_labels) in enumerate(target_loader):
                _labels=_labels.to(self.device).long()
                _list_of_indices,_segments_ids = _list_of_indices.to(self.device), _segments_ids.to(self.device)
                _output = self.model(_list_of_indices,_segments_ids)
                _loss=self.criterion(_output,_labels)
                loss_history.append(_loss.detach().item())
                _,_prediction= torch.max(_output, 1)
                predictions=np.append(predictions,_prediction.data.to('cpu'))
                labels=np.append(labels,_labels.cpu())
                torch.cuda.empty_cache()
                gc.collect()
            cm=ConfusionMatrix(labels,predictions)
        torch.cuda.empty_cache()
        gc.collect()
        return cm,np.mean(loss_history)

    def train(self,model_config,train_loader,valid_loader,epochs=100,print_every=100,validate_at_epoch=0):
        model=self.model
        train_res=np.array([])
        train_lbl=np.array([])
        if(not self.check_point_loaded):
            self.loss_history=[]
            self.test_loss_history=[]
            self.learning_rate=[]
            self.cm_test=[]
            self.cm_train=[]
            self.last_epoch=0
        elif(self.check_point_loaded):
            self.loss_history=self.checkpoint['train_loss_history']
            self.test_loss_history=self.checkpoint['test_loss_history']
            self.learning_rate=self.checkpoint['learning_rate_history']
            self.cm_test=self.checkpoint['cm_test']
            self.cm_train=self.checkpoint['cm_train']
            self.last_epoch=self.checkpoint['last_epoch']
        self.train_loops=len(train_loader)//print_every
        for e in range(self.last_epoch,self.epochs,1):
            self.e=e

            for i,(list_of_indices,segments_ids,labels) in enumerate(train_loader):
                model.train()
                list_of_indices,segments_ids,labels=list_of_indices.to(self.device),segments_ids.to(self.device),labels.to(self.device)
                output=model(list_of_indices,segments_ids)
                loss=self.criterion(output,labels)
                self.loss_history.append(loss.data.item())
                self.learning_rate.append(self.scheduler.get_lr())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                _,prediction= torch.max(output, 1)  
                train_res=np.append(train_res,(prediction.data.to('cpu')))
                train_lbl=np.append(train_lbl,labels.data.cpu().numpy())
                if((i+1)%print_every==0):
                    cm=ConfusionMatrix(train_lbl,train_res)
                    self.cm_train.append(cm)
                    print("epoch: ",e+1," step: ",(i+1)//print_every,"/",self.train_loops)
                    print("Batch Loss: ",np.mean(self.loss_history[len(self.loss_history)-print_every:len(self.loss_history)-1]))
                    print('train results: \n')
                    self.print_results(cm)
                    train_res=np.array([])
                    train_lbl=np.array([])
                torch.cuda.empty_cache()
                gc.collect()

            print("epoch: ",e+1,"Train  Loss: ",np.mean(self.loss_history[-1*(len(train_loader)-1):]),"\n")

            if(((e+1)>=validate_at_epoch)):
                print("************************")
                print("validation started ...","\n")
                _cm,_loss=self.predict(valid_loader)
                self.test_loss_history.append(_loss)
                print('test loss: ', _loss)
                self.print_results(_cm)
                print("************************","\n")
                self.cm_test.append(_cm)
            self.save_it(self.save_folder)
            self.scheduler.step()        
            
