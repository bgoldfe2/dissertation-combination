# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023

from Model_Config import Model_Config

from evaluate import evaluate_all_models
from driver import get_parser
from ensemble import averaging, max_vote

def test_eval(args: Model_Config):

    evaluate_all_models(args)

def test_average_ensemble(args):

    return averaging(args)

def test_max_vote(args):

    max_vote(args)

if __name__=="__main__":
    
    parser = get_parser()
    raw_args = parser.parse_args()

    # Declare the model list and pre-trained model
    model_list = [0, 1, 2, 4 , 5]
    pretrained_model = 'roberta-base'
        
    args = Model_Config(raw_args)
    args.model_list = model_list
    args.pretrained_model = pretrained_model

    # TODO currently hardcode this test run folder
    run2test =  "2023-08-14_16_20_29--roberta-base" #2023-07-03_14_53_05--deberta-v3-base"
    folder_name = "../Runs/" + run2test 

    # High level folders defined
    args.run_path=folder_name
    args.model_path = folder_name + "/Models/"
    args.output_path = folder_name + "/Output/"
    args.figure_path = folder_name  + "/Figures/"
    args.ensemble_path = folder_name  + "/Ensemble/"

    print('args.model_path in eval_test are\n',args.model_path)

    # Test the evaluate.py - evaluate_all_models() function
    # This I guess passed as I can do a full training run 8/14
    test_eval(args)

    # Test the averaging() function in ensembles.py
    #avg_rst = test_average_ensemble(args)
    #print(type(avg_rst))
    #print(avg_rst)

    # Test the max_vote() function in ensembles.py
    #test_max_vote(args)