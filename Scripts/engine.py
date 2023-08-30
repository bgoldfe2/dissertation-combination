# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023
# adapted from prior work

import torch
import torch.nn as nn
#from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

# import utils
import utils
from tqdm.auto import tqdm

from Model_Config import Model_Config

def loss_fn(output, target):
    return nn.CrossEntropyLoss()(output, target)


def train_fn(data_loader, model, optimizer, device, scheduler, args: Model_Config):
    model.train()
    losses = utils.AverageMeter()
    progress_bar = tqdm(data_loader, total = len(data_loader))
    train_losses = []
    final_target = []
    final_output = []

    for ii, data in enumerate(progress_bar):
        # Get a single example of the input data to the model and stop
        #print(data)
        #asdf
        output, target, input_ids = generate_output(data, model, device, args)

        loss = loss_fn(output, target)
        #print("loss type is ", type(loss))
        #print("loss printed is ", loss)
        train_losses.append(loss.item())
        output = torch.log_softmax(output, dim = 1)
        output = torch.argmax(output, dim = 1)
       

        # BHG need to add in some print statements to understand ouput, target and input_ids
        # and to uncomment the modulus output during training below

        # if(ii%100 == 0 and ii!=0) or (ii == len(data_loader)-1):
        # print((f'ii={ii}, Train F1={f1},Train loss={loss.item()}, time={end-start}'))

        loss.backward() # Calculate gradients based on loss
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() # Adjust weights based on calculated gradients
        scheduler.step() # Update scheduler
        losses.update(loss.item(), input_ids.size(0))
        progress_bar.set_postfix(loss = losses.avg)
        final_target.extend(target.cpu().detach().numpy().tolist())
        final_output.extend(output.cpu().detach().numpy().tolist())
    f1 = f1_score(final_target, final_output, average='weighted')
    f1 = np.round(f1.item(), 4)
    return f1, np.mean(train_losses)

# This function performs one batch at a time e.g. batch = 32
def eval_fn(data_loader, model, device, args: Model_Config):
    model.eval()
    progress_bar = tqdm(data_loader, total = len(data_loader))
    val_losses = []
    final_target = []
    final_output = []

    with torch.no_grad():
        for ii, data in enumerate(progress_bar):
            output, target, input_ids = generate_output(data, model, device, args)

            loss = loss_fn(output, target)
            output = torch.log_softmax(output, dim = 1)
            
            output = torch.argmax(output, dim = 1)
            
            val_losses.append(loss.item())
            final_target.extend(target.cpu().detach().numpy().tolist())
            final_output.extend(output.cpu().detach().numpy().tolist())
    f1 = f1_score(final_target, final_output, average='weighted')
    f1 = np.round(f1.item(), 4)
    return f1, np.mean(val_losses)

# This function is called to evaluate the ? full test set?
def test_eval_fn(data_loader, model, device, args):
    #pretrained_model = args.pretrained_model
    model.eval()
    progress_bar = tqdm(data_loader, total = len(data_loader))
    val_losses = []
    final_target = []
    final_output = []

    # BHG this version adds in a final_probabilities for some reason as well as output
    final_probabilities = []

    with torch.no_grad():
        for ii, data in enumerate(progress_bar):
            output, target, input_ids = generate_output(data, model, device, args)

            loss = loss_fn(output, target)
            output = torch.log_softmax(output, dim = 1)
            regular_probs = torch.softmax(output, dim = 1)
            #print("output after log_softmax ", output)
            #print("output after regular softmax ", regular_probs)
            probabilities = output
            output = torch.argmax(output, dim = 1)
            #print("output after argmax ", output)
            val_losses.append(loss.item())
            final_target.extend(target.cpu().detach().numpy().tolist())
            final_output.extend(output.cpu().detach().numpy().tolist())
            final_probabilities.extend(probabilities.cpu().detach().numpy().tolist())
    print(f'Output length --- {len(final_output)}, Prediction length --- {len(final_target)}')
    return final_output, final_target, final_probabilities

def test_eval_fn_ensemble(data_loader, model, device, args):
    pretrained_model = args.pretrained_model
    model.eval()
    progress_bar = tqdm(data_loader, total = len(data_loader))
    val_losses = []
    final_target = []
    final_output = []

    with torch.no_grad():
        for ii, data in enumerate(progress_bar):
            output, target, input_ids = generate_output(data, model, device, args)

            loss = loss_fn(output, target)
            output = torch.log_softmax(output, dim = 1)
            output = torch.exp(output)
            val_losses.append(loss.item())
            final_target.extend(target.cpu().detach().numpy().tolist())
            final_output.extend(output.cpu().detach().numpy().tolist())
    return final_output, final_target

def generate_output(data, model, device, args: Model_Config):
        
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    target = data["target"]
    #print("TARGET in generate output for loss fn ", target)

    input_ids = input_ids.to(device, dtype = torch.long)
    attention_mask = attention_mask.to(device, dtype = torch.long)
    target = target.to(device, dtype=torch.long)

    model.zero_grad()

    output = model(input_ids=input_ids, attention_mask = attention_mask)
    #print("I got to output ", output)
    #print(" output softmax", torch.softmax(output, dim=1))
    

    return output, target, input_ids
