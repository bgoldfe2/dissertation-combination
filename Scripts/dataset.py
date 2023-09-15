# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023
# adapted from prior work

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer


class DatasetDeberta:
    def __init__(self, args, text, target):
        pretrained_model = args.pretrained_model
        print("in dataset_deberta this is pretrained model ", pretrained_model)
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.max_length = args.max_length
        self.target = target

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = "".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text = text,
            padding = "max_length",
            truncation = True,
            max_length = self.max_length
        )

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]

        return{
            "input_ids":torch.tensor(input_ids,dtype = torch.long),
            "attention_mask":torch.tensor(attention_mask, dtype = torch.long),
            "token_type_ids":torch.tensor(token_type_ids, dtype = torch.long),
            "target":torch.tensor(self.target[item], dtype = torch.long)
        }


class DatasetGPT_Neo:
    def __init__(self, args, text, target):
        pretrained_model = args.pretrained_model
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.tokenizer.pad_token = "[PAD]"
        self.max_length = args.max_length
        self.target = target

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = "".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text = text,
            padding = "max_length",
            truncation = True,
            max_length = self.max_length
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        return{
            "input_ids":torch.tensor(input_ids,dtype = torch.long),
            "attention_mask":torch.tensor(attention_mask, dtype = torch.long),
            "target":torch.tensor(self.target[item], dtype = torch.long)
        }

class DatasetGPT_Neo13:
    def __init__(self, args, text, target):
        pretrained_model = args.pretrained_model
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.tokenizer.pad_token = "[PAD]"
        self.max_length = args.max_length
        self.target = target

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = "".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text = text,
            padding = "max_length",
            truncation = True,
            max_length = self.max_length
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        return{
            "input_ids":torch.tensor(input_ids,dtype = torch.long),
            "attention_mask":torch.tensor(attention_mask, dtype = torch.long),
            "target":torch.tensor(self.target[item], dtype = torch.long)
        }

class DatasetRoberta:
    def __init__(self, args, text, target):
        pretrained_model = args.pretrained_model
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.max_length = args.max_length
        self.target = target

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = "".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text = text,
            padding = "max_length",
            truncation = True,
            max_length = self.max_length
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        return{
            "input_ids":torch.tensor(input_ids,dtype = torch.long),
            "attention_mask":torch.tensor(attention_mask, dtype = torch.long),
            "target":torch.tensor(self.target[item], dtype = torch.long)
        }

class DatasetAlbert:
    def __init__(self, args, text, target):
        pretrained_model = args.pretrained_model
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.max_length = args.max_length
        self.target = target

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = "".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text = text,
            padding = "max_length",
            truncation = True,
            max_length = self.max_length
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        return{
            "input_ids":torch.tensor(input_ids,dtype = torch.long),
            "attention_mask":torch.tensor(attention_mask, dtype = torch.long),
            "target":torch.tensor(self.target[item], dtype = torch.long)
        }

class DatasetXLNet:
    def __init__(self, args, text, target):
        pretrained_model = args.pretrained_model
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.max_length = args.max_length
        self.target = target

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = "".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text = text,
            padding = "max_length",
            truncation = True,
            max_length = self.max_length
        )

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]

        return{
            "input_ids":torch.tensor(input_ids,dtype = torch.long),
            "attention_mask":torch.tensor(attention_mask, dtype = torch.long),
            "token_type_ids":torch.tensor(token_type_ids, dtype = torch.long),
            "target":torch.tensor(self.target[item], dtype = torch.long)
        }
    
# TODO look into a split that maintains the balance for the trait classes
# new version has seed set to 7 not 42 and not to None which is in earlier version
def train_validate_test_split(df, train_percent=0.6, validate_percent=.2, seed=7):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

def train_validate_test_balanced_split(df, train_percent=0.6, validate_percent=.2, seed=7):
    np.random.seed(seed)
    #perm = np.random.permutation(df.index)
    a = df['target'].unique()
    trt_list = sorted(a)
    
    train = pd.DataFrame()
    validate = pd.DataFrame()
    test = pd.DataFrame()

    for trt in trt_list:
        df_trt=df.loc[df["target"] == trt]
        print('trait ', trt, ' length ', len(df_trt))
        perm = np.random.permutation(df_trt.index)
        m = len(df_trt.index)
        train_end = int(train_percent * m)
        validate_end = int(validate_percent * m) + train_end
        train = pd.concat([train, df.iloc[perm[:train_end]]])
        validate = pd.concat([validate, df.iloc[perm[train_end:validate_end]]])
        test = pd.concat([test, df.iloc[perm[validate_end:]]])

        print('length of test is ', len(train))
    
    return train, validate, test

if __name__=="__main__":
    full_df = pd.read_csv("../Dataset/Full/not_full.csv")
    train, valid, test = train_validate_test_balanced_split(full_df)

    a = train['target'].unique()
    trt_list = sorted(a)

    for tgt_cls in trt_list:
        print(''.join(['class ', str(tgt_cls), ' length is ', str(len(train.loc[train['target'] == tgt_cls]))]))