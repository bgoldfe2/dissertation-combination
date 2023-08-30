# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023
# adapted from prior work

from os import name
import pandas as pd
import torch
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score


from evaluate import test_evaluate, calc_roc_auc
from engine import test_eval_fn_ensemble, test_eval_fn

from utils import sorting_function, evaluate_ensemble, print_stats, load_prediction, set_device, load_models, generate_dataset_for_ensembling

def max_vote(args):
    print(f'\n---Max voting ensemble---\n')

    deberta, xlnet, roberta, albert, gptneo = load_prediction(args)

    target = []
    deberta_pred = []
    xlnet_pred = []
    roberta_pred = []
    albert_pred = []
    gptneo_pred = []

    for index in range(len(deberta)):
       target.append(deberta['target'][index])
       deberta_pred.append(deberta['y_pred'][index])
       xlnet_pred.append(xlnet['y_pred'][index])
       roberta_pred.append(roberta['y_pred'][index])
       albert_pred.append(albert['y_pred'][index])
       gptneo_pred.append(gptneo['y_pred'][index])

    max_vote_df = pd.DataFrame()
    max_vote_df['target'] = target
    max_vote_df['deberta'] = deberta_pred
    max_vote_df['xlnet'] = xlnet_pred
    max_vote_df['roberta'] = roberta_pred
    max_vote_df['albert'] = albert_pred
    max_vote_df['gptneo'] = gptneo_pred

    # print_stats(max_vote_df, deberta, xlnet, roberta, albert)
    # BHG addtional lines into this function until line 88
    preds = []

    for index in range(len(max_vote_df)):
        values = max_vote_df.iloc[index].values[1:]
        sorted_values = sorted(Counter(values).items(), key = sorting_function, reverse=True)
        preds.append(sorted_values[0][0])
        
    max_vote_df['pred'] = preds

    print("In max_vote going to evaluate_ensemble")
    evaluate_ensemble(max_vote_df, args)
    
def max_vote3():
    print(f'\n---Max voting ensemble for the best three classifiers---\n')

    deberta, xlnet, roberta, albert, gptneo = load_prediction()

    target = []
    
    xlnet_pred = []
    roberta_pred = []
    albert_pred = []
    gptneo_pred = []

    for index in range(len(deberta)):
       target.append(deberta['target'][index])
       
       xlnet_pred.append(xlnet['y_pred'][index])
       roberta_pred.append(roberta['y_pred'][index])
       albert_pred.append(albert['y_pred'][index])
       gptneo_pred.append(gptneo['y_pred'][index])

    max_vote_df = pd.DataFrame()
    max_vote_df['target'] = target
    
    max_vote_df['xlnet'] = xlnet_pred
    max_vote_df['roberta'] = roberta_pred
    max_vote_df['albert'] = albert_pred
    max_vote_df['gptneo'] = gptneo_pred

    # print_stats(max_vote_df, deberta, xlnet, roberta, albert)
    # end of additional lines ? what changed?
    preds = []

    for index in range(len(max_vote_df)):
        values = max_vote_df.iloc[index].values[1:]
        sorted_values = sorted(Counter(values).items(), key = sorting_function, reverse=True)
        preds.append(sorted_values[0][0])
        
    max_vote_df['pred'] = preds

    evaluate_ensemble(max_vote_df)

# BHG Added new function
def rocauc(args):
    deberta, xlnet, roberta, albert, gptneo = load_models()
    test_df = pd.read_csv(f'{args.dataset_path}test.csv').dropna()
    device = set_device()

    deberta.to(device)
    test_data_loader = generate_dataset_for_ensembling(args, df =test_df)
    y_pred, y_test, y_proba = test_eval_fn(test_data_loader, deberta, device, pretrained_model="microsoft/deberta-v3-base")
    calc_roc_auc(np.array(y_test), np.array(y_proba), args, name='DEBERTA')
    del deberta, test_data_loader

    xlnet.to(device)
    test_data_loader = generate_dataset_for_ensembling(args, df=test_df)
    y_pred, y_test, y_proba = test_eval_fn(test_data_loader, xlnet, device, args)
    calc_roc_auc(np.array(y_test), np.array(y_proba), args, name='XLNet')
    del xlnet, test_data_loader

    roberta.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="roberta-base", df=test_df)
    y_pred, y_test, y_proba = test_eval_fn(test_data_loader, roberta, device, pretrained_model="roberta-base")
    calc_roc_auc(np.array(y_test), np.array(y_proba), args, name='RoBERTa')
    del roberta, test_data_loader

    albert.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="albert-base-v2", df=test_df)
    y_pred, y_test, y_proba = test_eval_fn(test_data_loader, albert, device, pretrained_model="albert-base-v2")
    calc_roc_auc(np.array(y_test), np.array(y_proba), args, name='albert')
    del albert, test_data_loader

    gptneo.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="EleutherAI/gpt-neo-125m", df=test_df)
    y_pred, y_test, y_proba = test_eval_fn(test_data_loader, gptneo, device, pretrained_model="EleutherAI/gpt-neo-125m")
    calc_roc_auc(np.array(y_test), np.array(y_proba), args, name='GPTNEO')
    del gptneo, test_data_loader
    
    

def averaging(args):
    all_trait_models = load_models(args)
    
    
    #deberta, xlnet, roberta, albert, gptneo = load_models(args)
    test_df = pd.read_csv(f'{args.dataset_path}test.csv').dropna()
    device = set_device(args)

    deberta.to(device)
    args.pretrained_model="microsoft/deberta-v3-base"
    test_data_loader = generate_dataset_for_ensembling(args, df =test_df)
    deberta_output, target = test_eval_fn_ensemble(test_data_loader, deberta, device, args)
    del deberta, test_data_loader

    xlnet.to(device)
    args.pretrained_model="xlnet-base-cased"
    test_data_loader = generate_dataset_for_ensembling(args, df=test_df)
    xlnet_output, target = test_eval_fn_ensemble(test_data_loader, xlnet, device, args)
    del xlnet, test_data_loader

    roberta.to(device)
    args.pretrained_model="roberta-base"
    test_data_loader = generate_dataset_for_ensembling(args, df=test_df)
    roberta_output, target = test_eval_fn_ensemble(test_data_loader, roberta, device, args)
    del roberta, test_data_loader

    albert.to(device)
    args.pretrained_model="albert-base-v2"
    test_data_loader = generate_dataset_for_ensembling(args, df=test_df)
    albert_output, target = test_eval_fn_ensemble(test_data_loader, albert, device, args)
    del albert, test_data_loader
    # BHG a lot of extra code in here?
    gptneo.to(device)
    args.pretrained_model="EleutherAI/gpt-neo-125m"
    test_data_loader = generate_dataset_for_ensembling(args, df=test_df)
    gptneo_output, target = test_eval_fn_ensemble(test_data_loader, gptneo, device, args)
    del gptneo, test_data_loader

    #gptneo13.to(device)
    #test_data_loader = generate_dataset_for_ensembling(pretrained_model="EleutherAI/gpt-neo-1.3B", df=test_df)
    #gptneo_output, target = test_eval_fn_ensemble(test_data_loader, gptneo, device, pretrained_model="EleutherAI/gpt-neo-1.3B")
    #del gptneo, test_data_loader

    # Create Averaging-Ensemble dictionary and store the results
    avg_ens_results = {}
    
    print(deberta_output)
    print(gptneo_output)
    print('------------------------------')
    output1 = np.add(deberta_output, xlnet_output)
    output2 = np.add(roberta_output, albert_output)
    output = np.add(output1, output2)
    output = np.add(output, gptneo_output)
    output = (np.divide(output,5.0))
    output = np.argmax(output, axis=1)

    # Results for test (truth) and predicted (inference)
    y_test = target
    y_pred = output
    avg_ens_results.update({
    "test": y_test,
    "pred": y_pred
    })
    
    print(f'\n---Probability averaging ensemble---\n')
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    avg_ens_results.update({
        "Accuracy": acc,
        "matthews_corrcoef": mcc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })
    
    print('Accuracy:', acc)
    print('Mcc Score:', mcc)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1_score:', f1)
    print('classification_report: ', classification_report(y_test, y_pred, digits=4))
    


    conf_mat = confusion_matrix(y_test,y_pred)
    print(conf_mat)
    avg_ens_results.update({
        "conf_mat": conf_mat
    })

    return avg_ens_results
    
# TODO this method is for the three model variant used by prior implementors
def averaging3(args):
    xlnet, roberta, gptneo = load_models()
    test_df = pd.read_csv(f'{args.dataset_path}test.csv').dropna()
    device = set_device()

    

    xlnet.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="xlnet-base-cased", df=test_df)
    xlnet_output, target = test_eval_fn_ensemble(test_data_loader, xlnet, device, pretrained_model="xlnet-base-cased")
    del xlnet, test_data_loader

    roberta.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="roberta-base", df=test_df)
    roberta_output, target = test_eval_fn_ensemble(test_data_loader, roberta, device, pretrained_model="roberta-base")
    del roberta, test_data_loader

    

    gptneo.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="EleutherAI/gpt-neo-125m", df=test_df)
    gptneo_output, target = test_eval_fn_ensemble(test_data_loader, gptneo, device, pretrained_model="EleutherAI/gpt-neo-125m")
    del gptneo, test_data_loader
    
    output1 = np.add(gptneo_output, xlnet_output)
    output2 = np.add(roberta_output, output1)
    
    output = (np.divide(output2,3.0))
    output = np.argmax(output, axis=1)

    y_test = target
    y_pred = output
    
    print(f'\n---Probability averaging ensemble---\n')
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
    


    conf_mat = confusion_matrix(y_test,y_pred)
    print(conf_mat)