# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023

import pandas as pd
import numpy as np
from Model_Config import rev_traits
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def make_cm_plot(cm, run_folder):
    print(cm)

    fig1, ax = plt.subplots(figsize=(14, 12))
    #title_font = {'size':'18'}
    plt.autoscale()
    #plt.title("CM for MultiLabel", fontdict=title_font)
    sns.heatmap(cm, annot=True, fmt="g", annot_kws={'size': 24}, cbar_kws={'ticks': []})
    label_font = {'size':'18'} 
    # Oooops switched labels to correct 
    #plt.ylabel('Predicted', fontdict=label_font)
    #plt.xlabel('True', fontdict=label_font)
    trt_labels = rev_traits.keys()
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_yticklabels(trt_labels)
    ax.set_xticklabels(trt_labels)
    plt.show()  # Show all plots at the end - can be same for saving?
    
    fig1.savefig(''.join([run_folder, 'cm.png']))

if __name__=="__main__":
    test_run = '../Runs/'
    cm = [  [1538, 6, 9, 4, 8, 6], \
            [7, 1567, 3, 3, 3, 3], \
            [32,30, 1471, 21, 35, 24], \
            [22, 36, 36, 1420, 38, 32],\
            [31, 40, 28, 26, 1430, 30], \
            [6, 6, 6, 8, 5, 1571]]
        
    make_cm_plot(cm, test_run)
