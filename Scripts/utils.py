# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023
# adapted from prior work

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from model import DeBertaFGBC, RobertaFGBC, XLNetFGBC, AlbertFGBC, GPT_NeoFGBC, GPT_Neo13FGBC
from dataset import DatasetDeberta, DatasetRoberta, DatasetXLNet, DatasetAlbert, DatasetGPT_Neo, DatasetGPT_Neo13

import os
from datetime import datetime
from Model_Config import Model_Config, traits
from glob import glob
from collections import defaultdict
import copy

#os.chdir("/home/bruce/dev/dissertation-final/Scripts")
#print("this is the folder??? ",os.getcwd())

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_folders(args: Model_Config) -> Model_Config:
    # Create the Runs Experiment folder location for Model,s Output, Figures
    # Get current time, remove microseconds, replace spaces with underscores
    current_datetime = str(datetime.now().replace(microsecond=0)).replace(" ","_")
    
    # NOTE: for multi-architecture runs this run will append only the first model type
    #print("error",args.pretrained_model)
    folder_name = "../Runs/" + current_datetime.replace(':','_') + "--" + args.pretrained_model #.split('/',1)[1] not needed roberta-base is HF model
    n=7 # number of letters in Scripts which is the folder we should be running from
    cur_dir = os.getcwd()
    #print(cur_dir)
    #print('folder name ', folder_name)
    
    # Parse out any subfolders for model descriptors e.g. microsoft/DeBERTa
    foo = args.model_list
    subfolders = []
    for bar in foo:
        if '/' in bar:
            fubar = bar.split('/',1)[0]
            subfolders.append(fubar)
    print("model name subfolders if any ", subfolders)

    ensemble_subfolders = ['/Models/', '/Output/', '/Figures/']

    # High level folders defined
    fld = ['/Models/', '/Output/', '/Figures/', '/Ensemble/']
    args.model_path = folder_name + "/Models/"
    args.output_path = folder_name + "/Output/"
    args.figure_path = folder_name  + "/Figures/"
    args.ensemble_path = folder_name + "/Ensemble/"
    #print('args.model_path are\n',args.model_path)
    
    if cur_dir[len(cur_dir)-n:] != 'Scripts':
        print('Run driver.py from Scripts Directory')        
    else:
        # Make the parent folder for this run
        os.mkdir(folder_name)

        # Create the subfolders as needed for models
        top_list = []
        for top in fld:
            fld_name = folder_name + top
            print(fld_name)
            top_list.append(fld_name)
            os.mkdir(fld_name)
        for sub in subfolders:
            for top in top_list:
                sub_name = top + sub + '/'
                print(sub_name)
                os.mkdir(sub_name)
        for ens in ensemble_subfolders:
            ens_folder = folder_name + "/Ensemble/" + ens
            print(ens_folder)
            os.mkdir(ens_folder)
            

    #print('args type ', type(args))
    #print('args.model path value ', args.model_path)

    return args

def set_device(args):
    device = ""
    if(args.device=="cpu"):
        device = "cpu"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if(device=="cpu"):
            print("GPU not available.")
    return device

def sorting_function(val):
    return val[1]    

def get_output_results(args)->dict[str, str]:
    file_dict = dict()
    mod_list = args.model_list
    for mod in mod_list:
        dir_results = args.output_path + '*' + mod + '*'
        result = glob(dir_results)[0]
        file_dict[mod]=result
    print(mod_list)
    print(file_dict)
    return file_dict

def load_prediction(args):

    # TODO in future this needs to be a loop not repeated code
    
    file_map = get_output_results(args)
    
    #print(type(file_map))
    search_key = 'deberta'
    deberta_path = [val for key, val in file_map.items() if search_key in key][0]
    #print(deberta_path)
    deberta = pd.read_csv(deberta_path)
    #print(deberta.shape)
    #print(deberta.head())
    
    search_key= 'xlnet'
    xlnet_path = [val for key, val in file_map.items() if search_key in key][0]
    #print(xlnet_path)
    xlnet = pd.read_csv(xlnet_path)
    #print(xlnet.shape)
    #print(xlnet.head())
        
    search_key= 'roberta'
    roberta_path = [val for key, val in file_map.items() if search_key in key][0]
    #print(roberta_path)
    roberta = pd.read_csv(roberta_path)
    #print(roberta.shape)
    #print(roberta.head())
    
    search_key= 'albert'
    albert_path = [val for key, val in file_map.items() if search_key in key][0]
    #print(albert_path)
    albert = pd.read_csv(albert_path)
    #print(albert.shape)
    #print(albert.head())
    
    search_key= 'gpt-neo'
    gpt_neo_path = [val for key, val in file_map.items() if search_key in key][0]
    #print(gpt_neo_path)
    gpt_neo = pd.read_csv(gpt_neo_path)
    #print(gpt_neo.shape)
    #print(gpt_neo.head())
    
    return deberta, xlnet, roberta, albert, gpt_neo

def print_stats(max_vote_df, deberta, xlnet, roberta, albert):
    print(max_vote_df.head())
    print(f'---Ground Truth---\n{deberta.target.value_counts()}')
    print(f'---DeBerta---\n{deberta.y_pred.value_counts()}')
    print(f'---XLNet---\n{xlnet.y_pred.value_counts()}')
    print(f'---Roberta---\n{roberta.y_pred.value_counts()}')
    print(f'---albert---\n{albert.y_pred.value_counts()}')

def evaluate_ensemble(max_vote_df, args):
    y_test = max_vote_df['target'].values
    y_pred = max_vote_df['pred'].values
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print('Accuracy:', acc)
    print('Mcc Score:', mcc)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1_score:', f1)
    print('classification_report: ', classification_report(y_test, y_pred, digits=4))
    
    max_vote_df.to_csv(f'{args.output_path}Ensemble-{args.ensemble_type}---test_acc---{acc}.csv', index = False)

    conf_mat = confusion_matrix(y_test,y_pred)
    print(conf_mat)

def generate_dataset_for_ensembling(args, df):
    if(args.pretrained_model == "microsoft/deberta-v3-base"):
        dataset = DatasetDeberta(args, text=df.text.values, target=df.target.values)
    elif(args.pretrained_model== "roberta-base"):
        dataset = DatasetRoberta(args, text=df.text.values, target=df.target.values)
    elif(args.pretrained_model== "xlnet-base-cased"):
        dataset = DatasetXLNet(args, text=df.text.values, target=df.target.values)
    elif(args.pretrained_model == "albert-base-v2"):
        dataset = DatasetAlbert(args, text=df.text.values, target=df.target.values)
    elif(args.pretrained_model == "EleutherAI/gpt-neo-125m"):
        dataset = DatasetGPT_Neo(args, text=df.text.values, target=df.target.values)
    elif(args.pretrained_model == "EleutherAI/gpt-neo-1.3m"):
        dataset = DatasetGPT_Neo13(args, text=df.text.values, target=df.target.values)

    data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = args.test_batch_size,
        shuffle = False
    )

    return data_loader

def load_models(args: Model_Config):

    # This function is refactored from multi-architecture to
    # multi-trait. The architecture for each model is consistent
    # as RoBERTa, but each of the five models are based on
    # one of the five cyberbullying traits
    # NOTE: I may want to add in a one vs rest model of Notcb vs all traits
    
    # TODO a path for each trait
    # TODO return a list of models - one model for each trait

    all_trt_models = defaultdict(list)
    print("traits is &&&&&&&&&&&&&& ", traits)
    lm_traits = copy.deepcopy(traits)
    lm_traits.pop('3')
    just_trts = list(lm_traits.values())

    print('just trt list of values ', just_trts)

    # Loop required to capture each of the current (variable for future growth) five trait models
    for trt in just_trts:
        mdl_path = ''.join([args.model_path, trt, '_Best_Val_Acc.bin'])
        print('model path in load models is ', mdl_path)

        roberta_path = (mdl_path)
        args.pretrained_model="roberta-base"
        roberta = RobertaFGBC(args)        
        roberta.load_state_dict(torch.load(roberta_path))
        all_trt_models[trt] = roberta
    
    #print("hope this works ", all_trt_models)
    return all_trt_models 

def oneHot(arr):
    b = np.zeros((arr.size, arr.max()+1))
    b[np.arange(arr.size),arr] = 1
    return b

