# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023

import pandas as pd
import os

os.chdir("/home/bruce/dev/dissertation-one-v-rest/Scripts/")
#os.chdir("/content/drive/MyDrive/Github/dissertation-one-v-rest/Scripts/")
def parse_all_traits():

    fld_path = '../Dataset/SixClass/'
    six = ['test.csv', 'train.csv', 'valid.csv']
    num_trt = 6

    rtn = []

    # Iterate over three files for all six traits per file
    for flnm in six:
        # Read in the file
        six_df = pd.read_csv(''.join([fld_path,flnm]))
        inner = []
        
        # Filter for each trait
        for trt in range(num_trt):
            six_trait=six_df.loc[six_df['target'] == trt]
            inner.append(six_trait)
        rtn.append(inner)
    
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

    traits ={ '0': "Age", '1': "Ethnicity", '2': "Gender", '3': "Notcb", '4': "Others", '5': "Religion"}
    split = ["val","train","test"]
    #val, train, test = parse_by_trait(4)
    #print("num train ",len(train), " val ", len(val), " test ", len(test))

    fld_bin_path ='../Dataset/Binary/'
    all = parse_all_traits()
    for ids,df in enumerate(all):
        print('file ', split[ids], ' is ', len(df), ' traits')
        for i,trt_num in enumerate(df):
            # Drop original and second indices and reset new one per file
            trt_num.drop('Unnamed: 0', axis=1, inplace=True)
            trt_num.reset_index(inplace=True, drop=True)
            print(type(trt_num))
            print(trt_num.head())
            
            
            # save the csv file
            trt_num.to_csv(''.join([fld_bin_path,split[ids],'//',split[ids],'_',traits.get(str(i)),'.csv'])\
                           , index=False, header=True)
