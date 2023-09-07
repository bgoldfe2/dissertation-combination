# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023

import pandas as pd
import numpy as np
from Model_Config import traits
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def get_results(run_folder):
    # Flag that identifies output results
    results_flag = '_acc-'
    results_file_list = []
    output_folder = ''.join([run_folder,'Output/'])
    print(output_folder)
    for filename in os.listdir(output_folder):
        f = os.path.join(output_folder, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if results_flag in filename:
                results_file_list.append(filename)

    

    return results_file_list

def parse_sbe(run_folder):
################# Outer Loop to read in all the files aka traits in Ensemble/Output and loop #################################

    results_file_list = get_results(run_folder)                
    print(results_file_list)
    
    for results in results_file_list:    
        # file trait pair during loop
        t1 = results.split('_')[0]
        t2 = results.split('_')[1].split('-')[0]
        print('The file traits this iteration are ',t1, ' and ',t2)

        # Set the file location for one v one combo runs        
        file = ''.join([run_folder,'Output/',results])

        df = pd.read_csv(file)
        total_all = len(df)
        print(total_all)
        print(df.columns)
        # Number in per label that are correct
        df['match'] = df['target']==df['y_pred']
        
        # Number in per label that are correct
        count = df.groupby('label').size()
        print("count is of type ", type(count))
        print("keys are ", count.axes)
        print("get value for Age ", count.get('Age'))
        print("Size of each trait \n", count)

        # Aggregate Confusion Matrix generation for the model        
        print("confusion matrix for ",t1," versus ", t2)
        cm = confusion_matrix(df['target'], df['y_pred'])
        print("the type of the confusion matrix is ", type(cm))
        # Print the confusion matrix
        print(cm)

        fig1, ax = plt.subplots()
        plt.title(''.join(["Trait ", t1, " vs ", t2]))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        trt_labels = [t1,t2]
        ax.set_yticklabels(trt_labels)
        ax.set_xticklabels(trt_labels)
        #plt.show()  # Show all plots at the end - can be same for saving?
        fig1.savefig(''.join([run_folder, 'Ensemble/Figures/','ensemble-bin-OvO-',t1,'-vs-', t2, '-conf-mat.pdf']))
        
        # Check to see the numbers add to 9541 = size of the test set
        cm_sum =  cm.sum()
        print("sum of numbers in cm is ", cm_sum) 
        
        df['false_pos'] = np.where(df['target']==0, 1, 0) & np.where(df['y_pred']==1, 1, 0)
        df['false_neg'] = np.where(df['target']==1, 1, 0) & np.where(df['y_pred']==0, 1, 0)
        df_cnt_fp = df.groupby('label')['false_pos'].apply(lambda x: (x==True).sum()).reset_index(name='count')
        df_cnt_fn = df.groupby('label')['false_neg'].apply(lambda x: (x==True).sum()).reset_index(name='count')
        
        print('false positives')
        print(df_cnt_fp)
        print(type(df_cnt_fp))
        print(df_cnt_fp.axes)

        print("traits in loop are ", t1, " vs ", t2)

        fp = df_cnt_fp.loc[df_cnt_fp['label']==t1, 'count'].values[0]
        fn = df_cnt_fn.loc[df_cnt_fn['label']==t1, 'count'].values[0]
        print(type(fp))
        print(fp)
        
        print('false negatives')
        print(df_cnt_fn)

        # Create each sub-confusion matrix of 2 x 2 for the five traits
        # Test hard coded for religion
        total_t1 = count.get(t1)  # 1575
        print("total in ", t1, " is ", total_t1)

        total_true_t1 = cm[0][0]
        total_true_t2 = cm[1][1]
        
        cm_trt = np.array([[total_true_t1, fp], [fn, total_true_t2]])
        
        print(cm_trt)

        # Show the distribution of false cyberbullying inferences that should have been Notcb
        
        fig2, ax = plt.subplots()
        # Create a barplot
        x = df_cnt_fp.iloc[:, 0].to_list()
        y = df_cnt_fp.iloc[:, 1].to_list()
        #y = cm_religion.T[1]
        print(" x ", x, " y ", y)
        plt.title("".join([t1, " vs ", t2, " with " , t1, " that were labelled ", t2]))
        plt.bar(x, y)
        fig2.savefig(''.join([run_folder, 'Ensemble/Figures/','ensemble-bin-OvO',t1,'-False-', t2, '-bar-plot.pdf']))


        # Show the distribution of the Notcb inferences that should have been cyberbullying
        fig3, ax = plt.subplots()
        # Create a barplot
        x = df_cnt_fn.iloc[:, 0].to_list()
        y = df_cnt_fn.iloc[:, 1].to_list()
        #y = cm_religion.T[1]
        print(" x ", x, " y ", y)
        plt.title("".join([t1, " vs ", t2, " Model, ",t2, " and all the rest that were labelled ", t1]))
        plt.bar(x, y)
        fig3.savefig(''.join([run_folder, 'Ensemble/Figures/','ensemble-bin-', t2, '-False-',t1,'.pdf']))

        # Show the plot TURN ON FOR DEBUG
        #plt.show()
        
        

if __name__=="__main__":
    test_run = '../Runs/2023-09-01_17_11_29--roberta-large/'
    #semantic_vote(test_run)
        
    parse_sbe(test_run)