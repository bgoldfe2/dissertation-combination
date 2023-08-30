# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023

traits ={ '0': "Age", '1': "Ethnicity", '2': "Gender", '3': "Notcb", '4': "Others", '5': "Religion"}

class Model_Config:

    def __init__(self, args):
        
        self.max_length=args.max_length         
        self.train_batch_size=args.train_batch_size    
        self.valid_batch_size=args.valid_batch_size     
        self.test_batch_size=args.test_batch_size      
        self.epochs=args.epochs
        self.learning_rate=args.learning_rate
        self.weight_decay=args.weight_decay
        self.adamw_epsilon=args.adamw_epsilon
        self.warmup_steps=args.warmup_steps
        self.classes=args.classes
        self.dropout=args.dropout
        self.seed=args.seed
        self.device=args.device
        self.pretrained_model=args.pretrained_model
        self.roberta_hidden=args.roberta_hidden
        
        # Needs to be updated in Version 3 for tree ensemble
        self.ensemble_type=args.ensemble_type

        self.run_path=args.run_path
        self.dataset_path=args.dataset_path
        self.model_path=args.model_path
        self.output_path=args.output_path
        self.figure_path=args.figure_path
        self.ensemble_path=args.ensemble_path
        self.split=args.split

        self.model_list = None