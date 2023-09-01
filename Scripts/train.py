# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023
# adapted from prior work

import pandas as pd;
import numpy as np;
import torch
from transformers import AdamW, get_scheduler
from collections import defaultdict

import engine
from model import RobertaFGBC, XLNetFGBC, AlbertFGBC, GPT_NeoFGBC, DeBertaFGBC, GPT_Neo13FGBC
from dataset import DatasetRoberta, DatasetXLNet, DatasetAlbert, DatasetGPT_Neo, DatasetDeberta, DatasetGPT_Neo13
from evaluate import test_evaluate

import utils
from visualize import save_acc_loss_curves   #save_acc_curves, save_loss_curves 
from dataset import train_validate_test_split

import utils
import matplotlib.pyplot as plt
from Model_Config import Model_Config, traits
import os
from copy import deepcopy

#os.chdir('/home/bruce/dev/dissertation-one-v-rest/Scripts')

def get_trt_from_pair(tp):
    return tp.split('_')[0], tp.split('_')[1]

def get_combo_data(t1, t2, df_in):
    #print("df_in is ", type(df_in))
    #print(df_in)
    df1 = df_in.loc[df_in['label']==t1,:].copy()
    df2 = df_in.loc[df_in['label']==t2,:].copy()

    #print("df1 after loc filter ", df1)

    # This is required for the CrossEntropy function to work correctly
    # This will require decoding during the ensemble function
    df1['target'] = 0
    df2['target'] = 1

    #print("df2 is ", type(df2))
    #print(df2)

    df_rtn = pd.concat([df1, df2])

    return df_rtn

def run(args: Model_Config):
    print("This is the model name ", args.pretrained_model)
    print("type that args is in run method ", type(args))
    print("This is the args.dataset_path in train run method", args.dataset_path)
    
    # Create the Combination of Six Traits pairs set of models 15 models (C6,2)
    comb_trt = ['Age_Ethnicity', 'Age_Gender', 'Age_Notcb', 'Age_Others', 'Age_Religion',
     'Ethnicity_Gender', 'Ethnicity_Notcb', 'Ethnicity_Others', 'Ethnicity_Religion', 'Gender_Notcb',
     'Gender_Others', 'Gender_Religion', 'Notcb_Others', 'Notcb_Religion', 'Others_Religion']
    

    for trt_pair in comb_trt:
        train_df_full = pd.read_csv(''.join([args.dataset_path, 'train.csv'   ])).dropna()
        valid_df_full = pd.read_csv(''.join([args.dataset_path, 'valid.csv'   ])).dropna()
        test_df_full = pd.read_csv(''.join([args.dataset_path, 'test.csv'   ])).dropna()

        t1, t2 = get_trt_from_pair(trt_pair)
        
        print(t1, ' --- ', t2)
        train_df = get_combo_data(t1, t2, train_df_full)
        valid_df = get_combo_data(t1, t2, valid_df_full)
        test_df = get_combo_data(t1, t2, test_df_full)
        

        # NOTE Text encoding occurs at model instantiation
        # Create the dataset classes for train, valid, and test  
        train_dataset = generate_dataset(train_df, args)
        train_data_loader = torch.utils.data.DataLoader(
            dataset = train_dataset,
            batch_size = args.train_batch_size,
            shuffle = True
        )

        valid_dataset = generate_dataset(valid_df, args)
        valid_data_loader = torch.utils.data.DataLoader(
            dataset = valid_dataset,
            batch_size = args.valid_batch_size,
            shuffle = True
        )

        test_dataset = generate_dataset(test_df, args)
        test_data_loader = torch.utils.data.DataLoader(
            dataset = test_dataset,
            batch_size = args.test_batch_size,
            shuffle = False
        )
        
        device = utils.set_device(args)

        model = set_model(args)
        # BHG model type and number of parameters initial instantiation
        print("Model Class: ", type(model), "Num Params: ",count_model_parameters(model))

        model = model.to(device)
        
        # BHG Model Paramter definition
        num_train_steps = int(len(train_df) / args.train_batch_size * args.epochs)

        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
             
        optimizer = AdamW(
            params = optimizer_parameters,
            lr = args.learning_rate,
            weight_decay = args.weight_decay,
            eps = args.adamw_epsilon
        )

        scheduler = get_scheduler(
            "linear",
            optimizer = optimizer,
            num_warmup_steps = num_train_steps*0.2,
            num_training_steps = num_train_steps
        )

        print("---Starting Training---")
        
        history = defaultdict(list)
        best_acc = 0.0
        
        for epoch in range(args.epochs):
            print(f'Epoch {epoch + 1}/{args.epochs}')
            print('-'*10)

            train_acc, train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler, args)
            print(f'Epoch {epoch + 1} --- Training loss: {train_loss} Training accuracy: {train_acc}')
            val_acc, val_loss = engine.eval_fn(valid_data_loader, model, device, args)
            print(f'Epoch {epoch + 1} --- Validation loss: {val_loss} Validation accuracy: {val_acc}')
            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            # SAVE MODEL if best so far going through epochs
            if val_acc>best_acc:
                print(f'Epoch {epoch + 1} val_acc {val_acc} best_acc {best_acc} trait {trt_pair}')
                torch.save(model.state_dict(), f"{args.model_path}{trt_pair}_Best_Val_Acc.bin")
                # BHG needed to set best_acc to val_acc this was missing in prior implementation
                best_acc=val_acc

            
        print(f'\n---History---\n{history}')
        print("##################################### Testing ############################################")
        pred_test, acc = test_evaluate(trt_pair,test_df, test_data_loader, model, device, args)
        #pred_test.to_csv(f'{args.output_path}{trt_pair}---test_acc---{acc}.csv', index = False)

        # Create and save the Accuracy and Loss plot during epoch training per trait
        plt_acc_loss = save_acc_loss_curves(args, trt_pair, history)
        
        del model, train_data_loader, valid_data_loader, train_dataset, valid_dataset
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("##################################### Task End ############################################")    
    
def create_dataset_files(args):
    if args.dataset == "FGBC":
        df = pd.read_csv(f'{args.dataset_path}dataset.csv').dropna()

        if args.classes == 5:
            indexnames = df[ df['label'] == 'Notcb' ].index
            df = df.drop(indexnames , inplace=False)
            df = df.reset_index()
            df.loc[df['target']==5, "target"] = 3
        print("This is the length of df in FIVE ",len(df))
    elif args.dataset == "Twitter":
        df = pd.read_csv(f'{args.dataset_path}twitter_dataset.csv').dropna()

    #Splitting the dataset
    train_df, valid_df, test_df = train_validate_test_split(df)
    train_df.to_csv(f'{args.dataset_path}train.csv')
    valid_df.to_csv(f'{args.dataset_path}valid.csv')
    test_df.to_csv(f'{args.dataset_path}test.csv')


def generate_dataset(df, cur_args: Model_Config):
    if(cur_args.pretrained_model == "microsoft/deberta-v3-base"):
        return DatasetDeberta(text=df.text.values, target=df.target.values, args=cur_args)
    elif(cur_args.pretrained_model == "EleutherAI/gpt-neo-125m"):
        return DatasetGPT_Neo(text=df.text.values, target=df.target.values, args=cur_args)
    elif(cur_args.pretrained_model == "EleutherAI/gpt-neo-1.3B"):
        return DatasetGPT_Neo13(text=df.text.values, target=df.target.values, args=cur_args)
    elif(cur_args.pretrained_model== "roberta-base"):
        return DatasetRoberta(text=df.text.values, target=df.target.values, args=cur_args)
    elif(cur_args.pretrained_model== "xlnet-base-cased"):
        return DatasetXLNet(text=df.text.values, target=df.target.values, args=cur_args)
    elif(cur_args.pretrained_model == "albert-base-v2"):
        return DatasetAlbert(text=df.text.values, target=df.target.values, args=cur_args)
    

def set_model(args):
    # BHG debug
    print("The model in the args is ", args.pretrained_model)
    
    if(args.pretrained_model == "microsoft/deberta-v3-base"):
        return DeBertaFGBC(args)
    elif(args.pretrained_model == "EleutherAI/gpt-neo-125m"):
        return GPT_NeoFGBC(args)
    elif(args.pretrained_model == "EleutherAI/gpt-neo-1.3B"):
        return GPT_Neo13FGBC(args)
    elif(args.pretrained_model == "roberta-base"):
        return RobertaFGBC(args)
    elif(args.pretrained_model == "xlnet-base-cased"):
        return XLNetFGBC(args)
    elif(args.pretrained_model == "albert-base-v2"):
        return AlbertFGBC(args)

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

