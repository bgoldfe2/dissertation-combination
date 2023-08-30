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