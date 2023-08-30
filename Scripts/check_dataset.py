import pandas as pd
import numpy as np
import os

label = '/test/'
folder = ''.join(['../Dataset/Binary', label])


total = 0
for file in os.listdir(folder):
    
    print(file)

    df = pd.read_csv(''.join([folder, file]))
    print(np.where(pd.isnull(df)))
    length = len(df)
    total+=length
    print("length of full dataset is ", length)
    print(df.isna().sum())
    print(df.isnull().sum())

print("total records in ", label, " is ",total)
