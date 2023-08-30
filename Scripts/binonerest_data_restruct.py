# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023

import pandas as pd
import copy
from Model_Config import traits

def parse_all_traits():

    fld_path = '../Dataset/SixClass/'
    six = ['test.csv', 'train.csv', 'valid.csv']
    #num_trt = 6

    rtn = []

    # Iterate over three files for all six traits per file
    for flnm in six:
        # Read in the file
        six_df = pd.read_csv(''.join([fld_path,flnm]))
                        
        rtn.append(six_df)
    
    return rtn

            
def parse_by_trait(tgt):

    fld_path = '../Dataset/SixClass/'
    six = ['test.csv', 'train.csv', 'valid.csv']
    
    rtn = []

    # Iterate over three files for all six traits per file
    for flnm in six:
        # Read in the file
        six_df = pd.read_csv(''.join([fld_path,flnm]))
        print('file ', flnm, ' is shape ', six_df.shape)
        print(six_df.head())
        
        six_trait=six_df.loc[six_df['target'] == tgt]
        print('    trait ', tgt, 'is shape ',six_trait.shape)
        print(six_trait.head())
        print()
        rtn.append(six_trait)

    return rtn[0], rtn[1], rtn[2]

def make_trait_based_files(tgt):
    fld_path = '../Dataset/SixClass/'
    six = ['test.csv', 'train.csv', 'valid.csv']
    
    rtn = []

    # Iterate over three files for all six traits per file
    for flnm in six:
        # Read in the file
        six_df = pd.read_csv(''.join([fld_path,flnm]))
        #print('file ', flnm, ' is shape ', six_df.shape)
        #print(six_df.head())
        
        six_trait=six_df.loc[six_df['target'] == tgt]
        #print('    trait ', tgt, 'is shape ',six_trait.shape)
        #print(six_trait.head())
        #print()
        rtn.append(six_trait)

    return rtn[0], rtn[1], rtn[2]


if __name__=="__main__":

    split = ["test","train","val"]    

    fld_bin_path ='../Dataset/BinOneRest/'
    all = parse_all_traits()
    print(type(all[0]))
        
    # dataset_type is the train, val, test set of data
    for dataset_type in range(3):
        df = all[dataset_type]
        df.drop('Unnamed: 0', axis=1, inplace=True)
        df.reset_index(inplace=True, drop=True)
        for i in range(6):           
            df_copy = copy.deepcopy(df) 
            df_copy.loc[df_copy['target'] != i, 'target'] = 6
            df_copy.loc[df_copy['target'] == i, 'target'] = 0
            df_copy.loc[df_copy['target'] == 6, 'target'] = 1
            
            print(df_copy.head(15))
            # save the csv file
            print(''.join([fld_bin_path,split[dataset_type],'/',split[dataset_type],'_',traits.get(str(i)),'_one_v_rest.csv']))
            #af
            df_copy.to_csv(''.join([fld_bin_path,split[dataset_type],'/',split[dataset_type],'_',traits.get(str(i)),'_one_v_rest.csv'])\
                           , index=True, header=True)
            print('saved the file ',''.join([fld_bin_path,split[dataset_type],'/',split[dataset_type],'_',traits.get(str(i)),'_one_v_rest.csv']))
              