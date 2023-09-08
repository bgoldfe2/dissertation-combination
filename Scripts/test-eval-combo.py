# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023

from Model_Config import Model_Config

from combo_evaluate import evaluate_all_combo_models, eval_vote_files, eval_vote_files_permissive
from driver import get_parser

def test_combo(run2test):
    parser = get_parser()
    raw_args = parser.parse_args()

    # Declare the model list and pre-trained model
    comb_trt_pairs = ['Age_Ethnicity', 'Age_Gender', 'Age_Notcb', 'Age_Others', 'Age_Religion',
     'Ethnicity_Gender', 'Ethnicity_Notcb', 'Ethnicity_Others', 'Ethnicity_Religion', 'Gender_Notcb',
     'Gender_Others', 'Gender_Religion', 'Notcb_Others', 'Notcb_Religion', 'Others_Religion']
    pretrained_model = 'roberta-large'
        
    args = Model_Config(raw_args)
    args.model_list = comb_trt_pairs
    args.pretrained_model = pretrained_model

    
    # High level folders defined
    args.run_path=run2test
    args.model_path = run2test + "/Models/"
    args.output_path = run2test + "/Output/"
    args.figure_path = run2test  + "/Figures/"
    args.ensemble_path = run2test  + "/Ensemble/"

    print('args.model_path in eval_test are\n',args.model_path)

    # Perform inference for test data
    evaluate_all_combo_models(args)


if __name__=="__main__":
    # TODO currently hardcode this test run folder
    test_folder =  '../Runs/2023-09-01_17_11_29--roberta-large'

    #test_combo(test_folder)

    # HARDCODE add in args for later implementations
    ensemble_path = ''.join([test_folder, '/Ensemble/'])

    #eval_vote_files(ensemble_path)
    eval_vote_files_permissive(ensemble_path)