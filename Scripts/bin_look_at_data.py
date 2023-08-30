import pandas as pd
from Model_Config import traits

test_data_path = '../Dataset/Binary/test/'

df = pd.read_csv(''.join([test_data_path, 'test_Age.csv']))
print(df.head())

traits.pop('3')
just_cb = traits.values()

all_test_data = []

for trt in just_cb:
    all_test_data.append(pd.read_csv(''.join([test_data_path, 'test_', trt, '.csv'])))

df_all = pd.concat(all_test_data, axis=0).reset_index()



print(df_all)