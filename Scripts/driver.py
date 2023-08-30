# Utility to drive models and ensemble combinations
# Bruce Goldfeder
# CSI 999, George Mason University
# Dec 27, 2022

import argparse
import warnings
import torch
import utils
import numpy as np
from train import run
from Model_Config import Model_Config

from evaluate import evaluate_all_models
from ensemble import averaging

# Suppress copious PyTorch warnings output
warnings.filterwarnings("ignore")

# Kludge for onsite network
# import os
# import ssl
# os.environ['CURL_CA_BUNDLE'] = '/etc/pki/ca-trust/extracted/openssl/ca-bundle.trust.crt'
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
# End Kludge onsite network

# New v3.0 - Converts to Binary model per trait for six trait, six models
def train_all_models(my_args: Model_Config):
    # Iterate through trait models all of a single architecture default is RoBERTa
    print("type of my_args in train_all_models ", type(my_args))
    print("model list type ", my_args.model_list)

    for i in my_args.model_list:
        my_args.pretrained_model = i
        run(my_args)

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--max_length", default=128, type=int,  help='Maximum number of words in a sample')
    parser.add_argument("--train_batch_size", default=32, type=int,  help='Training batch size')
    parser.add_argument("--valid_batch_size", default=32, type=int,  help='Validation batch size')
    parser.add_argument("--test_batch_size", default=32, type=int,  help='Test batch size')
    parser.add_argument("--epochs", default=4, type=int,  help='Number of training epochs')
    parser.add_argument("-lr","--learning_rate", default=2e-5, type=float,  help='The learning rate to use')
    parser.add_argument("-wd","--weight_decay", default=1e-4, type=float,  help=' Decoupled weight decay to apply')
    parser.add_argument("--adamw_epsilon", default=1e-8, type=float,  help='AdamW epsilon for numerical stability')
    parser.add_argument("--warmup_steps", default=0, type=int,  help='The number of steps for the warmup phase.')
    parser.add_argument("--classes", default=2, type=int, help='Number of output classes set to 2 for binary sensors')
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--device", type=str, default="gpu", help="Training device - cpu/gpu")
    #parser.add_argument("--dataset", type=str, default="FGBC", help="Select Dataset - FGBC/Twitter")

    parser.add_argument("--pretrained_model", default="roberta-base", type=str, help='Name of the pretrained model')  
    parser.add_argument("--roberta_hidden", default=768, type=int, help='Number of hidden states for Roberta')

    # Need to change for Version 3 to tree ensemble
    parser.add_argument("--ensemble_type", type=str, default="max-voting", help="Ensemble type - max-voting or averaging")

    parser.add_argument("--run_path", default="../Runs/", type=str, help='Path to Run logs')
    parser.add_argument("--dataset_path", default="../Dataset/BinOneRest/", type=str, help='Path to dataset file')
    parser.add_argument("--model_path", default="../Models/", type=str, help='Save best model')
    parser.add_argument("--output_path", default="../Output/", type=str, help='Get predicted labels for test data')
    parser.add_argument("--figure_path", default="../Figures/", type=str, help='Directory for accuracy and loss plots')
    parser.add_argument("--ensemble_path", default="../Ensemble/", type=str, help='Directory for accuracy and loss plots, model, and figures for Ensemble')
    parser.add_argument("--split", default="no", type=str, help='If base file needs to be splitted into Train, Val, Test')

    return parser



if __name__=="__main__":

    # Parse command line arguments and defaults
    # This is being deprecated to the Model_Config class
    parser = get_parser()
    raw_args = parser.parse_args()

    # Ensure repeatability in experiments with common seed values
    np.random.seed(raw_args.seed)
    torch.manual_seed(raw_args.seed)
    torch.cuda.manual_seed(raw_args.seed)
    
    # Declare the model list - From version 1 model ensemble
    #model_list = ['microsoft/deberta-v3-base', 'EleutherAI/gpt-neo-125m', 'roberta-base',\
    #                'xlnet-base-cased', 'albert-base-v2']
    
    # Version 3 - Single model in list being roberta-base
    #               Keeping list syntax as may include BertViz or others
    model_list = ['roberta-base']

    # convert immutable args to python class instance and set up dynamic folder structure
    args = Model_Config(raw_args)
    args.model_list = model_list
    my_args = utils.create_folders(args)

    # Test the Five Class run
    #args.classes = 2
    #args.dataset_path = "../Dataset/Binary/"
    #args.split = "yes"

    print("args type in driver main after create_folders ", type(args))
    train_all_models(args)
    # print("########################### TRAINING COMPLETE #########################################")
    # evaluate_all_models(args)
    # print("############################ EVALUATION COMPLETE ######################################")
    # averaging(args)
    # print("############################ ENSEMBLE COMPLETE ########################################")
