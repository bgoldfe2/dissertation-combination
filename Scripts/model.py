# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023
# adapted from prior work

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel 

# TODO Combine all these classes into one with conditional elements to differentiate

class DeBertaFGBC(nn.Module):
    def __init__(self, args):
        super().__init__()
        pretrained_model = args.pretrained_model
        self.DeBerta = AutoModel.from_pretrained(pretrained_model)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.deberta_hidden, 64)
        self.batch_norm = nn.LayerNorm(64)
        #self.drop2 = nn.Dropout(args.dropout)
        print("in the model init num classes is ", args.classes)
        self.out = nn.Linear(64, args.classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.DeBerta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            #output_hidden_states=True,
            return_dict=False
            #return_dict=True
        )
        #print(f'\n\nLast Hidden State Type - {type(last_hidden_state)}')
        #print(f'\n\nLast Hidden State Type - {last_hidden_state}')

        new_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        #mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        #print(f'\n\nLast New Hidden State Type - {type(new_last_hidden_state)}')
        #print(f'\n\nLast New Hidden State Size - {new_last_hidden_state.shape}')
        #print(f'\n\nLast New Hidden State Values - {new_last_hidden_state}')
        
        
        bo = self.drop1(new_last_hidden_state)
        #print(f'Dropout1 - {bo.shape}')
        bo = self.linear(bo)
        #print(f'Linear1 - {bo.shape}')
        bo = self.batch_norm(bo)
        #print(f'BatchNorm - {bo.shape}')
        bo = nn.Tanh()(bo)
        #bo = self.drop2(bo)
        #print(f'Dropout2 - {bo.shape}')

        # the return are the values of the last linear layer for each category
        output = self.out(bo)
        #print(f'Output - {output.shape}')
        return output

    def pool_hidden_state(self, last_hidden_state):
        tup_len = len(last_hidden_state)
        #print('Last hidden state shape - ',tup_len)
        tup_elem_lens = [len(a) for a in last_hidden_state]
        #print('last hidden state element shapes - ',tup_elem_lens)
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state    


class GPT_NeoFGBC(nn.Module):
    def __init__(self, args):
        super().__init__()
        pretrained_model = args.pretrained_model
        self.GPT2 = AutoModel.from_pretrained(pretrained_model)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.gpt_neo_hidden, 64)
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.GPT2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        
        bo = self.drop1(mean_last_hidden_state)
        bo = self.linear(bo)
        bo = self.batch_norm(bo)
        bo = nn.Tanh()(bo)
        bo = self.drop2(bo)

        output = self.out(bo)

        return output

    def pool_hidden_state(self, last_hidden_state):
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state

class GPT_Neo13FGBC(nn.Module):
    def __init__(self, args):
        super().__init__()
        pretrained_model = args.pretrained_model
        self.GPT2 = AutoModel.from_pretrained(pretrained_model)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.gpt_neo13_hidden, 64)
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.GPT2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        
        bo = self.drop1(mean_last_hidden_state)
        bo = self.linear(bo)
        bo = self.batch_norm(bo)
        bo = nn.Tanh()(bo)
        bo = self.drop2(bo)

        output = self.out(bo)

        return output

    def pool_hidden_state(self, last_hidden_state):
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state

class RobertaFGBC(nn.Module):
    def __init__(self, args):
        super().__init__()
        pretrained_model = args.pretrained_model
        self.Roberta = AutoModel.from_pretrained(pretrained_model)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.roberta_hidden, 64)
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)
        #print("num of classes in model init is ",args.classes)

    def forward(self, input_ids, attention_mask):
        _,last_hidden_state = self.Roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        bo = self.drop1(last_hidden_state)
        bo = self.linear(bo)
        bo = self.batch_norm(bo)
        bo = nn.Tanh()(bo)
        bo = self.drop2(bo)

        output = self.out(bo)

        return output

class AlbertFGBC(nn.Module):
    def __init__(self, args):
        super().__init__()
        pretrained_model = args.pretrained_model
        self.albert = AutoModel.from_pretrained(pretrained_model)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.albert_hidden, 64)
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        
        bo = self.drop1(mean_last_hidden_state)
        bo = self.linear(bo)
        bo = self.batch_norm(bo)
        bo = nn.Tanh()(bo)
        bo = self.drop2(bo)

        output = self.out(bo)

        return output

    def pool_hidden_state(self, last_hidden_state):
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state

class XLNetFGBC(nn.Module):
    def __init__(self, args):
        super().__init__()
        pretrained_model = args.pretrained_model
        self.XLNet = AutoModel.from_pretrained(pretrained_model)
        self.drop1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.xlnet_hidden, 64)
        self.batch_norm = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(args.dropout)
        self.out = nn.Linear(64, args.classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        last_hidden_state = self.XLNet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)

        bo = self.drop1(mean_last_hidden_state)
        bo = self.linear(bo)
        bo = self.batch_norm(bo)
        bo = nn.Tanh()(bo)
        bo = self.drop2(bo)

        output = self.out(bo)

        return output
        
    def pool_hidden_state(self, last_hidden_state):
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state