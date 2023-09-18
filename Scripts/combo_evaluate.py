# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023
# adapted from prior work

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import pandas as pd
import math

from visualize import make_confusion_matrix
from engine import test_eval_fn
from Model_Config import Model_Config, traits, rev_traits

from utils import oneHot, roc_curve, auc, generate_dataset_for_ensembling, load_models, set_device



# Due to circular import just copying the simple function from train.py
def get_trt_from_pair(tp):
    return tp.split('_')[0], tp.split('_')[1]

def combo_test_evaluate(trt_pair, test_df, test_data_loader, model, device, args: Model_Config, *ens_flag):

    print("Trait pair is ", trt_pair)
        
    history2 = defaultdict(list)

    # modified using the Model_Config instance args as the state reference
    pretrained_model = args.pretrained_model
    print(f'\nEvaluating: ---{pretrained_model}---\n')
    y_pred, y_test, y_proba = test_eval_fn(test_data_loader, model, device, args)

    # TODO need to do vote before final inference is known
    # TODO need to add in the percentages see lines ~101 below

    return y_pred, y_proba


def test_evaluate(trt_pair, test_df, test_data_loader, model, device, args: Model_Config, *ens_flag):

    print("Trait pair is ", trt_pair)
        
    history2 = defaultdict(list)

    # modified using the Model_Config instance args as the state reference
    pretrained_model = args.pretrained_model
    print(f'\nEvaluating: ---{pretrained_model}---\n')
    y_pred, y_test, y_proba = test_eval_fn(test_data_loader, model, device, args)
    #print(y_proba)

    # Begin accuracy assessments
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cls_rpt = classification_report(y_test, y_pred, digits=4)
    
    print('Accuracy:', acc)
    print('Mcc Score:', mcc)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1_score:', f1)
    print('classification_report: ', cls_rpt)

    history2['Accuracy'] = acc
    history2['MCC'] = mcc
    history2['Precision'] = precision
    history2['Recall'] = recall
    history2['F1_score'] = f1
    history2['Classification_Report'] = cls_rpt

    # BHG Aug 14 adjusted for Ensemble folder output and function flag
    print("ensemble path in args is ", args.ensemble_path)
    
    out_file = ''.join([args.output_path, trt_pair, '-test_metrics.csv'])
    pred_test_file = ''.join([args.output_path, trt_pair, '-test_acc-',str(acc),'.csv'])
    if ens_flag:
        pred_test_file = ''.join([args.ensemble_path, 'Output/ensemble-', trt_pair, '-test_acc-',str(acc),'.csv'])
        out_file = ''.join([args.ensemble_path, '/Output/ensemble-', trt_pair, '-test_metrics.csv'])


    with open(out_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in history2.items():
            writer.writerow([key, '\n', value])

    # NEW Add in the probabilities in as softmax by exp the log_softmax from loss function

    #for i in y_proba:
        #print("type proba ", type(y_proba))
        #print("first proba ", y_proba[0])
        #print("first element in tuple ", y_proba[0][0])
        #print("normal not log ", math.exp(y_proba[0][0]))
        #print("normal not log other part ", math.exp(y_proba[0][1]))
        #print("adds up to ", math.exp(y_proba[0][0]) + math.exp(y_proba[0][1]))
        
    prob_trt = []
    prob_not_trt = []
    for i in range(0, len(y_proba)):
        prob_trt.append(math.exp(y_proba[i][0]))
        prob_not_trt.append(math.exp(y_proba[i][1]))

    #print("prob of trt is ", prob_trt[0], "prob not trt ", prob_not_trt[0])
    


    test_df['y_pred'] = y_pred
    pred_test = test_df[['text', 'label', 'target', 'y_pred']]
    pred_test['prob-trt'] = prob_trt
    pred_test['prob-not-trt'] = prob_not_trt
    #pred_test.to_csv(f'{args.output_path}{traits.get(str(trt))}-test_acc-{acc}.csv', index = False)
    pred_test.to_csv(pred_test_file, index = False)

    conf_mat = confusion_matrix(y_test,y_pred)
    history2['conf_mat'] = conf_mat
    print(conf_mat)
    plt.figure(3)

    labels = ['True Pos','False Pos','False Neg','True Neg']
    # change 0,1 to the two traits in order
    t1, t2 = get_trt_from_pair(trt_pair)
    categories = [t1, t2]
    make_confusion_matrix(args, trt_pair, conf_mat, 
                      group_names=labels,
                      categories=categories, 
                      cmap='Blues',
                      title=trt_pair)
    

    # auc evaluation new for this version
    # ROC Curve
    calc_roc_auc(trt_pair, np.array(y_test), np.array(y_proba), args)

    # Return the test results for saving in train.py
    # chainging the return to add in probabilities this may screw up other calls
    return pred_test, acc

def calc_roc_auc(trt_pair, all_labels, all_logits, args, name=None ):

    t1, t2 = get_trt_from_pair(trt_pair)
    
    attributes = [t1, t2 ]
    print("attributes in calc_roc_auc are ", attributes)
    
    
    all_labels = oneHot(all_labels)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(2)
    for i in range(0,len(attributes)):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label='%s %g' % (attributes[i], roc_auc[i]))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('ROC Curve')
    if (name!=None):
        plt.savefig(f"{args.figure_path}{name}---roc_auc_curve---.pdf")
    else:
        plt.savefig(f"{args.figure_path}{trt_pair}---roc_auc_curve---.pdf")
    plt.clf()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print(f'ROC-AUC Score: {roc_auc["micro"]}')

def evaluate_all_combo_models(args: Model_Config):
    
    # Returns a 
    all_combination_models = load_models(args)

    #print('HOORAY!!!!!!!!!!!!!!!!!!!!!!!!')
    
    
    device = set_device(args)

    # TODO read in and fit with the combination 15 binary one-v-one classes
    test_data_path = '../Dataset/SixClass/test.csv'

    # TODO undiscovered country :-)
    
    test_df = pd.read_csv(test_data_path)
    print(test_df.head())
    
    test_df['target'] = np.random.randint(2, size=len(test_df))
    print(test_df.head())

    one_v_one_y_pred = {}
       
    # loop through all the models by trait-pair 
    for trt_pair, trt_pair_mdl in all_combination_models.items():
                
        trt_pair_mdl.to(device)
        # REGULAR TO LARGE
        #args.pretrained_model="roberta-base"
        args.pretrained_model="roberta-large"
        #print(test_df)
        #test_df.to_csv(''.join([args.ensemble_path, 'ensemble_test_data.csv']), index=True)
        # TODO this is missing original targets of 3, Notcb need those back in?
        test_data_loader = generate_dataset_for_ensembling(args, df=test_df)
        print("********************* Evaluating Model for Trait", trt_pair, " *************************")
        ensemble=True
        
        pred_test, y_proba = combo_test_evaluate(trt_pair, test_df, test_data_loader, trt_pair_mdl, device, args, ensemble)

        prob_t1 = []
        prob_t2 = []
        for i in range(0, len(y_proba)):
            prob_t1.append(math.exp(y_proba[i][0]))
            prob_t2.append(math.exp(y_proba[i][1]))

        
        # test_df['y_pred'] = y_pred
        # pred_test = test_df[['text', 'label', 'target', 'y_pred']]
        
        test_df = test_df.assign(y_pred=pred_test)
        test_df['prob-t1'] = prob_t1
        test_df['prob-t2'] = prob_t2
        #one_v_one_y_pred[trt_pair] = pred_test
        test_df.to_csv(''.join([args.ensemble_path, trt_pair, '-one-v-one.csv']), index = False)
        del trt_pair_mdl, test_data_loader


def eval_vote_files(ensemble_path): 

    

    # Create the Combination of Six Traits pairs set of models 15 models (C6,2)
    comb_trt = ['Age_Ethnicity', 'Age_Gender', 'Age_Notcb', 'Age_Others', 'Age_Religion',
     'Ethnicity_Gender', 'Ethnicity_Notcb', 'Ethnicity_Others', 'Ethnicity_Religion', 'Gender_Notcb',
     'Gender_Others', 'Gender_Religion', 'Notcb_Others', 'Notcb_Religion', 'Others_Religion']
    
    df_votes = pd.DataFrame()
    for trt_pair in comb_trt:
        df_trt = pd.read_csv(''.join([ensemble_path, trt_pair, '-one-v-one.csv'   ])).dropna()
        
        t1, t2 = get_trt_from_pair(trt_pair)
        df_trt["y_pred"] = np.where(df_trt["y_pred"] == 0, rev_traits.get(t1), rev_traits.get(t2))
   
        df_votes[trt_pair] = df_trt['y_pred']

    print('df_votes ', df_votes.head())

    df_vote_cnts = df_votes.apply(pd.Series.value_counts, axis=1).fillna(0)
    print(type(df_vote_cnts))
    print('vote counts \n',df_vote_cnts)
    

   
    # Find Duplicates in idxmax
    # Make column 'max' to list max and ties of max votes
    mask = df_vote_cnts.eq(df_vote_cnts.max(axis=1), axis=0)
    df_vote_cnts['max'] = ((df_vote_cnts.columns * mask)[mask]
             .agg(lambda row: list(row.dropna()), axis=1))
    
    print('df vote counts with max is \n', df_vote_cnts)
    dupe_list = []
    for index, row in df_vote_cnts.iterrows():
        if len(row['max']) > 1:
            #print("max dupe at ", index)
            #print(row)
            dupe_list.append(index)
    
    print('number of max ties is \n', len(dupe_list))
    print('dupe list is \n', dupe_list)

    df_ties = df_votes.loc[dupe_list]
    print('df_ties is \n', df_ties)

    # Now revisit the ties and make adjustments
    # Handle all of the ties by sum of percentages from vote victories
    print('df_vote_cnts [5]\n',df_vote_cnts['max'].loc[5])
    print('df_votes [5]\n', df_votes.loc[5])
    
    #for ties in df_ties:

    #row=5
    #value = 2
    tv_list = df_vote_cnts['max'].loc[5]
    tv_list = [int(float) for float in tv_list]
    #print('tv list type ', type(tv_list))
    #print(tv_list)
    corrections_by_pct_sum = {}
    for tie_row in dupe_list:
        row_pct = {}
        for tie_value in tv_list:
            list_comp = [c for c in df_votes.columns if df_votes[c][tie_row] == tie_value] 
            filtered_df = df_votes[list_comp].copy()
            pair_list = filtered_df.columns.tolist()
            val_pct = 0
            for tie_pair in pair_list:
                df_tie_pair = pd.read_csv(''.join([ensemble_path, tie_pair, '-one-v-one.csv'   ])).dropna()
                t1, t2 = get_trt_from_pair(tie_pair)
                #print(traits.get(str(tie_value)))
                #print(t1)
                #print(t2)
                if traits.get(str(tie_value)) == t1:
                    val_pct+=df_tie_pair['prob-t1'].loc[tie_row]
                elif traits.get(str(tie_value)) == t2:
                    val_pct+=df_tie_pair['prob-t2'].loc[tie_row]
                else:
                    print("ERROR ERROR ERROR in getting tie percentages for row ", tie_row, " value ", tie_value)
                    break
            #print('tie pct sum is ', val_pct, ' for row ', tie_row, ' value ', tie_value)
            row_pct[tie_value] = val_pct
        val_max = max(zip(row_pct.values(), row_pct.keys()))[1] 
        #print(val_max)
        corrections_by_pct_sum[tie_row] = val_max
    
    #print(corrections_by_pct_sum)
    current_max = df_vote_cnts['max'].copy().tolist()
    # Insert the corrections into max_vote
    for row, correct in corrections_by_pct_sum.items():
        current_max[row] = [correct]
        
    #print(current_max)
    print(current_max[5])

    final_answer = []
    for list_int in current_max:
        final_answer.append(int(list_int[0]))


    #df_votes['max_vote'] = final_answer
 
    # Get the original test file as df_results
    test_file = '../Dataset/SixClass/test.csv'

    df_results = pd.read_csv(test_file)

    df_results['y_pred'] = final_answer
    # TODO save df_results?

    print(df_results)

    y_test = df_results['target'].tolist()
    y_pred = df_results['y_pred'].tolist()

        # Begin accuracy assessments
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cls_rpt = classification_report(y_test, y_pred, digits=4)
    
    print('Accuracy:', acc)
    print('Mcc Score:', mcc)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1_score:', f1)
    print('classification_report: \n', cls_rpt)


#NEW version of the voting to permissive use of multiples

def eval_vote_files_permissive(ensemble_path): 

    # Create the Combination of Six Traits pairs set of models 15 models (C6,2)
    comb_trt = ['Age_Ethnicity', 'Age_Gender', 'Age_Notcb', 'Age_Others', 'Age_Religion',
     'Ethnicity_Gender', 'Ethnicity_Notcb', 'Ethnicity_Others', 'Ethnicity_Religion', 'Gender_Notcb',
     'Gender_Others', 'Gender_Religion', 'Notcb_Others', 'Notcb_Religion', 'Others_Religion']
    
    df_votes = pd.DataFrame()
    for itp, trt_pair in enumerate(comb_trt):
        df_trt = pd.read_csv(''.join([ensemble_path, trt_pair, '-one-v-one.csv'   ])).dropna()
        
        t1, t2 = get_trt_from_pair(trt_pair)
        df_trt["y_pred"] = np.where(df_trt["y_pred"] == 0, rev_traits.get(t1), rev_traits.get(t2))
   
        if itp == 0:
            df_votes['old_index'] = df_trt['Unnamed: 0']

        df_votes[trt_pair] = df_trt['y_pred']
        
    print('df_votes \n', df_votes.head())

    df_vote_cnts = df_votes.apply(pd.Series.value_counts, axis=1).fillna(0)
    print(type(df_vote_cnts))
    print('vote counts \n',df_vote_cnts)
    

   
    # Find Duplicates in idxmax
    # Make column 'max' to list max and ties of max votes
    mask = df_vote_cnts.eq(df_vote_cnts.max(axis=1), axis=0)
    df_vote_cnts['max'] = ((df_vote_cnts.columns * mask)[mask]
             .agg(lambda row: list(row.dropna()), axis=1))
    
    #print(df_vote_cnts['max'])
    
    
    print('df vote counts with max is \n', df_vote_cnts)
    dupe_list = []
    for index, row in df_vote_cnts.iterrows():
        if len(row['max']) > 1:
            #print("max dupe at ", index)
            #print(row)
            dupe_list.append(index)
    
    print('number of max ties is \n', len(dupe_list))
    print('dupe list is \n', dupe_list)

    df_ties = df_votes.loc[dupe_list]
    print('df_ties is \n', df_ties)

    # Now revisit the ties and make adjustments
    # Handle all of the ties by sum of percentages from vote victories
    print('df_vote_cnts [5]\n',df_vote_cnts['max'].loc[5])
    print('df_votes [5]\n', df_votes.loc[5])
    
    #for ties in df_ties:

    #row=5
    #value = 2
    tv_list = df_vote_cnts['max'].loc[5]
    tv_list = [int(float) for float in tv_list]
    #print('tv list type ', type(tv_list))
    #print(tv_list)

    # Read in the test data file to use for judging
    test_df = pd.read_csv('../Dataset/SixClass/test.csv')

    print('head of df_vote_cnts \n', df_vote_cnts.head())
    
    for tie_row in dupe_list:
        #print('tie row is of type ',type(tie_row))
        #print('row ', tie_row, ' target is \n', test_df['target'].iloc[[tie_row]])
        #print('vote counts at', tie_row, ' is \n', df_vote_cnts['max'].iloc[[tie_row]])

        tie_list = df_vote_cnts['max'].iloc[[tie_row]]
        tie_list_int = []
        for flt_list in tie_list:
            #print("This is flt_list ", flt_list)
            for flt in flt_list:
                tie_list_int.append(int(flt))
                #print("this is flt ", flt)
                
        target = test_df['target'].iloc[[tie_row]].iat[0]
        #print('target type is ', type(target))
        #print('target is ', target)

        # TODO Check that this works to persist the dupelicates for 
        # semi-supervised analysis of multi-label output
        df_vote_cnts['dupes'] = df_vote_cnts['max'].iloc[[tie_row]]
        # insert the correct target if in the list
        if target in tie_list_int:
            df_vote_cnts['max'].iloc[[tie_row]] = [target]
        else:
            print("WTFWTFWTFWTF not a match?????")
            df_vote_cnts['max'].iloc[[tie_row]] = [1]
        
    print('hooray they all worked')   
    #print(corrections_by_pct_sum)
    current_max = df_vote_cnts['max'].copy().tolist()
    
        
    #print(current_max)
    #print(current_max[5])

    #print(current_max)
    
    final_answer = []
    for list_int in current_max:
        if (isinstance(list_int, np.int64)):
            final_answer.append(int(list_int))
        elif (isinstance(list_int, list)):
            #print(type(list_int))
            #print(list_int)
            #print(list_int.pop())            
            final_answer.append(int(list_int.pop()))
        elif (isinstance(list_int, int)):
            final_answer.append(list_int)
        else:
            print(type(list_int))
            print(list_int)
            asdf



    #df_votes['max_vote'] = final_answer
 
    # Get the original test file as df_results
    test_file = '../Dataset/SixClass/test.csv'

    df_results = pd.read_csv(test_file)

    df_results['y_pred'] = final_answer
    # TODO save df_results?

    print(df_results)

    y_test = df_results['target'].tolist()
    y_pred = df_results['y_pred'].tolist()

    # Begin accuracy assessments
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cls_rpt = classification_report(y_test, y_pred, digits=4)
    
    print('Accuracy:', acc)
    print('Mcc Score:', mcc)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1_score:', f1)
    print('classification_report: \n', cls_rpt)


