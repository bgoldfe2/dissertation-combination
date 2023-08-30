# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023
# adapted from prior work

import matplotlib.pyplot as plt
from collections import defaultdict
from Model_Config import traits
import numpy as np
import seaborn as sns

# For Debug
#def save_acc_loss_curves(trt, history):
def save_acc_loss_curves(args, trt, history):

    plt.figure(1)
    plt.plot(range(1,5),history['train_acc'], label='train accuracy')
    plt.plot(range(1,5),history['val_acc'], label='validation accuracy')
    plt.title('Training and Validation Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Epoch')
    plt.xlim(1, 4)
    plt.xticks(range(1, 5))
    plt.plot(range(1,5),history['train_loss'], label='train loss')
    plt.plot(range(1,5),history['val_loss'], label='validation loss')
    plt.legend()
    #plt.ylim([0.0, 0.3])
    
    # For Debug
    #plt.savefig(f"{traits.get(str(trt))}---acc_loss---.pdf")
    plt.savefig(f"{args.figure_path}{traits.get(str(trt))}---acc_loss---.pdf")

    plt.clf()
    plt.close()


# Adapted from code at https://github.com/DTrimarchi10/confusion_matrix
# This code is upside down from regular depiction of confusion matrix (e.g. sklearn) and 
# required corrections in calling code and this funtion to correct
# calling code format:
# labels = ['True Pos','False Pos','False Neg','True Neg']
# categories = ['1', '0']
# make_confusion_matrix(args, trt, conf_mat, 
#                   group_names=labels,
#                   categories=categories, 
#                   cmap='Blues',
#                   title=traits.get(str(trt)))
def make_confusion_matrix(args, 
                          trt,
                          cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('Predicted label')
        plt.xlabel('True label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
    print("trait ", trt, "figure path", args.figure_path, " trait ", traits.get(str(trt)))
    #asdf
    plt.savefig(''.join([args.figure_path, traits.get(str(trt)), '_confusion_matrix.pdf']), dpi=400)
    plt.clf()
    plt.close()
    

if __name__=="__main__":
    
    history = defaultdict(list)
    history['train_acc'] = [0.8809, 0.9798, 0.9864, 0.991]
    history['val_acc'] = [0.9715, 0.9854, 0.9864, 0.9857]
    history['train_loss'] = [0.2564959865777443, 0.08274737073108554, 0.05754305351215104, 0.04171089528749387]
    history['val_loss'] = [0.11794130202157027, 0.057937183756042614, 0.055730998901782014, 0.058969416937819034]
    
    plot = save_acc_loss_curves(0, history)

    history = defaultdict(list)
    history['train_acc'] = [0.809, 0.8798, 0.9864, 0.991]
    history['val_acc'] = [0.7715, 0.7854, 0.7864, 0.7857]
    history['train_loss'] = [0.2564959865777443, 0.08274737073108554, 0.05754305351215104, 0.04171089528749387]
    history['val_loss'] = [0.11794130202157027, 0.057937183756042614, 0.055730998901782014, 0.058969416937819034]
   
    plot2 = save_acc_loss_curves(1, history)

    