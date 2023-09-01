# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023

from Model_Config import Model_Config

from combo_evaluate import evaluate_all_combo_models, eval_vote_files
from driver import get_parser

def test_combo():
    parser = get_parser()
    raw_args = parser.parse_args()

    # Declare the model list and pre-trained model
    comb_trt_pairs = ['Age_Ethnicity', 'Age_Gender', 'Age_Notcb', 'Age_Others', 'Age_Religion',
     'Ethnicity_Gender', 'Ethnicity_Notcb', 'Ethnicity_Others', 'Ethnicity_Religion', 'Gender_Notcb',
     'Gender_Others', 'Gender_Religion', 'Notcb_Others', 'Notcb_Religion', 'Others_Religion']
    pretrained_model = 'roberta-base'
        
    args = Model_Config(raw_args)
    args.model_list = comb_trt_pairs
    args.pretrained_model = pretrained_model

    # TODO currently hardcode this test run folder
    run2test =  "2023-08-30_18_54_29--roberta-base" 
    folder_name = "../Runs/" + run2test 

    # High level folders defined
    args.run_path=folder_name
    args.model_path = folder_name + "/Models/"
    args.output_path = folder_name + "/Output/"
    args.figure_path = folder_name  + "/Figures/"
    args.ensemble_path = folder_name  + "/Ensemble/"

    print('args.model_path in eval_test are\n',args.model_path)

    # Perform inference for test data
    evaluate_all_combo_models(args)


if __name__=="__main__":
    #test_combo()

    eval_vote_files()