# Estimating 3-SAT & #3-SAT with STS Configurations

## Required Python Libraries
```
itertools
re
pandas
numpy
torch
optuna
matplotlib
tqdm
sklearn
```


## File Structure

```
sts_config_3sat_pred
├── ganak
│   ├── include    # ganak dependencies
│   ├── lib        # ganak dependencies
│   └── ganak      # ganak executable file (generate ground truth for synthetic data)
├── sols_by_confs
│   ├── cnf_data
│   │   ├── test                       # 100k generated .cnf files & .csv file with solution and configuration counts for testing
│   │   ├── train                      # 5mil generated .cnf files & .csv file with solution and configuration counts for training
│   │   ├── val                        # 250k generated .cnf files & .csv file with solution and configuration counts for validation
│   │   ├── concat_training_data.py    # compile training data sub-jobs from SLURM
│   │   ├── gen_training_data.py       # driver for generating training data SLURM batch
│   │   ├── gen_training_data.sh       # SLURM script
│   │   └── generate.ipynb             # notebook file to generate .cnf files for testing/training/validation
│   ├── models
│   │   ├── model_validation           # collection of model architectures and hyperparamter tuning for validation
│   │   ├── classifier.py              # final classification model training script
│   │   ├── confusion_matrix_ttc.png   # final regression model training script
│   │   ├── regressor.py               # confusion matrix plot for classification model
│   │   └── roc_curve_ttc.png          # roc curve plot for classification model
│   └── utils                          # utility functions for generating data
└── README.md
```
### Important Note:
The project submission will not contain the number of `.cnf` files indicated above due to excessive size, instead a small sample will be provided in each directory. 

## Running Code
There are two categories of programs to run in this project to interact with models.
1. Run `python classifier.py` or `python regressor.py` to train the final tab-transformers and regernerate test data
2. Run `./model_validation/parameter_training.sh model_validation/<MODEL_FILE.py>` to run hyper-parameter tuning on any of the provided model architectures. !!! NOTE: This must be run from inside the `models` directory due to static paths !!!

Models in `/model_validation` are prefixed with `class_` or `reg_` to indicate the model as running classification (3-SAT) or regression (#3-SAT).

Another note: Due to unsatisfactory results, no models are saved permanently.








