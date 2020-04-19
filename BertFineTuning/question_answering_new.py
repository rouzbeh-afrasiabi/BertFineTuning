from BertFineTuning.utils import *
from BertFineTuning.question_answering_model_config import *
from transformers import BertTokenizer

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
from torch.autograd import Variable

import copy
import gc

from ast import literal_eval

from transformers import BertModel,BertConfig,BertForQuestionAnswering

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.insert(0, cwd)




random_state=123
torch.manual_seed(random_state)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)




class BertFineTuning():
    def __init__(self,model_config_user=model_config,base_model=BertModel,
                 base_tokenizer=BertTokenizer,base_pretrained_weights='bert-base-uncased',
                base_pretrained_config=BertConfig()):
        
        class View(nn.Module):
            def __init__(self, *shape):
                super(View, self).__init__()
                self.shape = shape
            def forward(self, input):
                return input.view(*self.shape)  
            
        class Split(nn.Module):
            def __init__(self, *split):
                super(Split, self).__init__()
                self.split = split
            def forward(self, input):
                return input.split(*self.split,dim=1)         
            
        class Network(nn.Module):
            def __init__(self, pre_trained_model,config):
                super().__init__()
               
                self.pre_trained_model=pre_trained_model.to(config['device'])
                self.pre_trained_out_features=list(list(self.pre_trained_model.children())[-1].children())[0].out_features
                self.model_output_features=config['max_token_length']
                self.classifier=nn.Sequential(OrderedDict([
                    ('fltn1',nn.Flatten()),
                    ('drp1',nn.Dropout(p=config['dropout_prob'])),
                    ('fc1', nn.Linear(self.pre_trained_out_features*self.model_output_features, self.model_output_features)),
                    ('bn_1',nn.BatchNorm1d(self.model_output_features)),
                    ('prelu1', nn.PReLU()),
                    ('fc2', nn.Linear(self.model_output_features, self.model_output_features)),
#                     ('rs1',View(-1, 2, self.model_output_features)),
#                     ('splt1',Split(1)),
                ]))
                
#                 self.flatten=nn.Flatten()
                
            def forward(self, tokens_tensor, segments_tensors):
                last_hidden_state, pooled_output = self.pre_trained_model(tokens_tensor, segments_tensors)
                logits = self.classifier(last_hidden_state)

                return (logits) 
        
            
        def __device():
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.check_point_loaded=False
        self.device = __device()
        self.criterion_config={}
        self.optimizer_config={}
        self.scheduler_config={}
        if(model_config_user):
            self.config=model_config_user
        else:
            self.config=model_config
        self.config['device']=self.device

        self._base_model=base_model(base_pretrained_config)
        self.base_pretrained_config=base_pretrained_config
        self._base_tokenizer=base_tokenizer
        self._base_pretrained_weights=base_pretrained_weights
        
        self.pre_trained_model=base_model.from_pretrained(base_pretrained_weights)
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
        self.scheduler_step_epoch=False
        self.scheduler_step_batch=False
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
        
    @staticmethod    
    def logits_to_one_hot(logits_start,logits_end):

        loc_start=torch.argmax(logits_start,dim=1).data.cpu().numpy()
        loc_end=torch.argmax(logits_end,dim=1).data.cpu().numpy()
        prediction=np.zeros_like(logits_end.detach().data.cpu().numpy())
        span=[list(range(*item)) for item in list(zip(loc_start,loc_end+1))] 
        for i,r in enumerate(span):
            r=[item for item in r if item<logits_end.shape[-1]]
            prediction[i,r]=1
        prediction=prediction.flatten()
        return(prediction)
    
    @staticmethod    
    def logit_to_one_hot(logit,threshold):

        return((torch.sigmoid(logit) > threshold).float())

    
    @staticmethod
    def index_to_labels_like(loc_start,loc_end,logit,is_flatten=True):
        loc_start=loc_start.data.cpu().numpy()
        loc_end=loc_end.data.cpu().numpy()
        labels=np.zeros_like(logit.detach().data.cpu().numpy())
        span=[list(range(*item)) for item in list(zip(loc_start,loc_end+1))] 
        for i,r in enumerate(span):
            r=[item for item in r if item<logit.shape[-1]]
            labels[i,r]=1
        if(is_flatten):
            labels=labels.flatten()
        return(labels)

    @staticmethod
    def span_to_labels_like(span,logit,is_flatten=True):
        labels=np.zeros_like(logit.detach().data.cpu().numpy()) 
        for i,r in enumerate(span):
            r=[item for item in literal_eval(r) if item<logit.shape[-1]]
            labels[i,r]=1
        if(is_flatten):
            labels=labels.flatten()
        return(labels)
    
    @staticmethod
    def exact_match(logits_start,logits_end,list_of_indices):
        loc_start=torch.argmax(logits_start,dim=1).data.cpu().numpy()
        loc_end=torch.argmax(logits_end,dim=1).data.cpu().numpy()
        span=[list(range(*item)) for item in list(zip(loc_start,loc_end+1))] 
        result=[]
        for i,r in enumerate(span):
            indices=list_of_indices[i,r].data.cpu().numpy()
#             tokens=base_tokenizer.convert_ids_to_tokens(indices)
#             string=base_tokenizer.convert_tokens_to_string(tokens)
            result.append(indices)
            
        return(result)
        
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
            test_res=np.array([])
            test_lbl=np.array([])
            loss_history=[0]
            labels=np.array([])
            
            for i,_loader_dict in  enumerate(target_loader):
                ids,list_of_indices,segments_ids,labels=_loader_dict.values()
                start_labels,end_labels,span=labels
                
                list_of_indices=list_of_indices.to(self.device)
                segments_ids=segments_ids.to(self.device)
                start_labels=start_labels.to(self.device).requires_grad_(False).long()
                end_labels=end_labels.to(self.device).requires_grad_(False).long()
                output_start,output_end=self.model(list_of_indices,segments_ids)

                start_loss=self.criterion(output_start,start_labels)                       
                end_loss=self.criterion(output_end,end_labels)
                loss=(start_loss+end_loss)/2
                loss_history.append(loss.detach().item())
                
                prediction=self.logits_to_one_hot(output_start,output_end)
                test_res=np.append(test_res,prediction)
                test_lbl=np.append(test_lbl,self.span_to_labels_like(span,output_start))
                

            cm=ConfusionMatrix(test_lbl,test_res)
        torch.cuda.empty_cache()
        gc.collect()
        return cm,np.mean(loss_history)

    def train(self,train_loader,valid_loader,epochs=100,print_every=100,validate_at_epoch=0):
        model=self.model
        train_res_start=np.array([])
        train_res_end=np.array([])
        train_res=np.array([])
        train_lbl_start=np.array([])
        train_lbl_end=np.array([])
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
            for i,_loader_dict in enumerate(train_loader):
                ids,list_of_indices,segments_ids,labels=_loader_dict.values()
                start_labels,end_labels,span=labels
                
                model.train()
                
                list_of_indices=list_of_indices.to(self.device)
                segments_ids=segments_ids.to(self.device)
                
                output=model(list_of_indices,segments_ids)
                labels=torch.from_numpy(self.span_to_labels_like(span,output,is_flatten=False)).requires_grad_(False).to(self.device)
                loss=self.criterion(output,labels)
                
                self.loss_history.append(loss.detach().data.cpu().numpy())
                self.learning_rate.append(self.scheduler.get_lr())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if(self.scheduler_step_batch):
                    self.scheduler.step()
                
                prediction=self.logit_to_one_hot(output,0.5)
                train_res=np.append(train_res,prediction.detach().data.cpu().numpy())
                train_lbl=np.append(train_lbl,labels.detach().data.cpu().numpy())
                
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
#             self.save_it(self.save_folder)        
                if(self.scheduler_step_epoch):
                    self.scheduler.step()       
