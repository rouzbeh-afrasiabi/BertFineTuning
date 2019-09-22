from BertFineTuning.utils import *
from BertFineTuning.data_config import *

import os
import sys
import math
import copy
import urllib
import zipfile
import pandas as pd
import requests
import shutil
import numpy as np
import matplotlib.pyplot as plt
from download import download
import re
from collections import OrderedDict
from sklearn.utils.class_weight import compute_class_weight
import spacy

import warnings

warnings.filterwarnings('ignore', message='Unverified HTTPS request')

cwd = str(os.getcwd())
sys.path.append(cwd)
sys.path.insert(0, cwd)

random_state=123


def create_folders(folders):
    for folder in folders:
        if(check_folder(folder)):
            pass
        else:
            print('*** Creating new folder named: ',folder) 
            os.mkdir(folder)
def download_files(files,download_folder):
    for file in files:
        [[_,location]]=file.items()
        file_name=os.path.basename(location)
        exists,_=check_file(file_name,download_folder)
        if(exists):
            pass
        else:
            print('*** Downloading : ',file_name)
            try:
                r = requests.get(location, auth=('usrname', 'password'), verify=False,stream=True)
                r.raw.decode_content = True   
                with open(os.path.join(download_folder,file_name), 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
            except:
                raise Exception('Failed')
            
#folders
data_folder=os.path.join(cwd,'data')
download_folder=os.path.join(cwd,'download')
processed_data_folder=os.path.join(cwd,'data','processed')
original_data_folder=os.path.join(cwd,'data','original')
cleaned_data_folder=os.path.join(cwd,'data','cleaned')
create_folders([data_folder,download_folder,processed_data_folder,cleaned_data_folder]) 

#files

dataset={'dataset':"https://github.com/rouzbeh-afrasiabi/PublicDatasets/raw/master/train.csv.zip"}
word_vectors={"en_vectors_web_lg":"https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz"}
toDownload=[dataset,word_vectors]
download_files(toDownload,download_folder)
try:
    nlp = spacy.load("en_vectors_web_lg")
except:
    os.system("pip install ./download/en_vectors_web_lg-2.1.0.tar.gz")
    nlp = spacy.load("en_vectors_web_lg")


def remove_stop_words(doc,nlp):
    output_string=[]
    for token in doc:
        if (not token.is_stop):
            output_string.append(token.text)
        else:
            pass
    new_doc=nlp(" ".join(output_string))
    return(doc)

def fix_oov(doc,nlp):
    #spell check was removed because it affects the results
#     from spellchecker import SpellChecker
    output=[]
#     spell = SpellChecker()
    for token in doc:
        new_token=''
        if(token.text in nlp.vocab):
            new_token=token.text
#         elif(token.text!=spell.correction(token.text)):
#             new_token=spell.correction(token.text)
#             print(token.text,' spelling changed to: ',new_token)
        else:
          new_token=replace_punct(token.text)
#           print(token.text,"not found in vocabulary, changed to: ",new_token)
        if(new_token):
            output.append(new_token)
    new_doc=nlp(" ".join(output))
    return(new_doc)
  
def lemmatize(doc,nlp):
    output_string=[]
    for token in doc:
        output_string.append(token.lemma_)
    new_doc=nlp(" ".join(output_string))
    return(new_doc)

def remove_punct(doc,nlp):
    output_string=[]
    for token in doc:
        if((not token.is_punct) and (not token.is_space)):
            output_string.append(token.lemma_)
    new_doc=nlp(" ".join(output_string))
    return(new_doc)


def to_index(doc):
    ids=[]
    for token in doc:
        if token.has_vector:
            id = nlp.vocab.vectors.key2row[token.norm]
        else:
            id = None 
        ids.append(id)
    return(ids)

def replace_punct(input_string):
    import string
    output=input_string.translate(str.maketrans(string.punctuation,
                 ' ' * len(string.punctuation))).replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ').strip()
    
    return(output)
def remove_numbers(doc,nlp):
    output_string=[]
    for token in doc:
        if(not token.is_digit):
            output_string.append(token.lemma_)
    new_doc=nlp(" ".join(output_string))
    return(new_doc)  

def to_vec(doc,nlp):
    vecs=[]
    for token in doc:
        if token.has_vector:
            vec = token.vector
        else:
            vec = None 
            print('empty vector generated!!')
        vecs.append(vec)
    return(np.array(vecs))

def to_piece(input):
    global nlp
    doc=nlp(input)
    global sp
    pieces=sp.EncodeAsPieces(doc.text)
    if(pieces):
        return(pieces)
    else:
        return(sp.EncodeAsPieces("<unk>"))
    
def process_doc(input_string):
    global _processed
    global nlp
    global _max_seq_len
    doc=nlp(str(input_string).lower())
    doc_oov=fix_oov(doc,nlp)
    doc_lemma=lemmatize(doc_oov,nlp)
    doc_stop=remove_stop_words(doc_lemma,nlp)
    doc_punct=remove_punct(doc_stop,nlp)
    if((_processed+1)%100000==0): 
        print('processed ',_processed+1, 'records')
    _processed=_processed+1

    return(doc_punct)


exists,_=check_file("train.csv",original_data_folder)

if(not exists):
    zip_file = zipfile.ZipFile(os.path.join(download_folder,"train.csv.zip"), 'r')
    zip_file.extractall(original_data_folder)
    zip_file.close()

_processed=0
exists,_=check_file('Main.csv',cleaned_data_folder)
if(not exists):
    train_df=pd.read_csv(os.path.join(original_data_folder,'train.csv'),encoding='utf-8',sep=',', engine='python')
    question1_clean=train_df.question1.apply(lambda x:process_doc(x))
    question2_clean=train_df.question2.apply(lambda x:process_doc(x))
    main_df=train_df.copy()
    main_df['question2']=question2_clean
    main_df['question1']=question1_clean
    main_df.to_csv(os.path.join(cleaned_data_folder,'Main.csv'))
else:
    train_df=pd.read_csv(os.path.join(cleaned_data_folder,'Main.csv'), index_col=[0])
    print('Loaded existing cleaned file')

train_df=train_df.dropna(axis=0,how='any')
weights=compute_class_weight('balanced',train_df['is_duplicate'].unique(),train_df['is_duplicate'].values.flatten())
weights_all=train_df['is_duplicate'].map(dict(zip(train_df['is_duplicate'].unique(),weights)))



def split_dataset(target_df, fracs, weights):
    global random_state
    datasets = []
    in_features = target_df.copy()
    for frac in fracs:
        temp = in_features.sample(frac=frac, replace=False,
                                  random_state=random_state,
                                  weights=weights)
        in_features.drop(temp.index, inplace=True)
        datasets.append(temp)
    return [data for data in datasets]

target_dataset = train_df
names = ['train_split', 'valid_split', 'test_split']

if not all([os.path.exists(os.path.join(cleaned_data_folder, name+'.csv'))
           for name in names]):
    split_data = split_dataset(target_dataset, [0.6, .5, 1],
                               weights_all)
    split_data_dict = dict(zip(names, [data for data in split_data]))
    for (i, data) in enumerate(split_data):
        data.to_csv(os.path.join(cleaned_data_folder, names[i]+'.csv'))
else:
    print ('Loading Saved file ...')
    split_data = []
    for (i, name) in enumerate(names):
        temp = pd.read_csv(os.path.join(cleaned_data_folder, name+'.csv'), index_col=[0])
        split_data.append(temp)
    split_data_dict = dict(zip(names, [data for data in split_data]))



bert_split_data = []
names = ['bert_train_split', 'bert_valid_split', 'bert_test_split']
if not all([os.path.exists(os.path.join(processed_data_folder, name+'.csv')) for name in names]):
    for (k, target) in copy.deepcopy(split_data_dict).items():
        target['question1'] = target['question1'].apply(lambda x:x[0:_max_string_length])
        target['question2'] = target['question2'].apply(lambda x:x[0:_max_string_length])
        marked_text = pd.DataFrame('[CLS] ' + target['question1']+ ' [SEP] ' + target['question2']+ ' [SEP] ', columns=['text'])
        marked_text['label'] = target['is_duplicate']
        bert_split_data.append(marked_text)
    
    for (i, data) in enumerate(bert_split_data):
        data.to_csv(os.path.join(processed_data_folder, names[i]+'.csv'))
# bert_split_data_dict = dict(zip(names, [data for data in bert_split_data]))





