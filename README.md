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
/Project-Root
|--/ganak
|  |--/include       # ganak dependencies
|  |--/lib           # ganak dependencies
|  |--ganak          # ganak executable file (generate ground truth for synthetic data)
|--/sols_by_confs
|  |--/cnf_data
|  |  |--/test       # 100k generated .cnf files & .csv file with solution and configuration counts for testing
|  |  |--/train      # 5mil generated .cnf files & .csv file with solution and configuration counts for training
|  |  |--/val        # 250k generated .cnf files & .csv file with solution and configuration counts for validation
|  |  |--concat_training_data.py    # compile training data sub-jobs from SLURM
|  |  |--gen_training_data.py       # driver for generating training data SLURM batch
|  |  |--gen_training_data.sh       # SLURM script
|  |  |--generate.ipynb             # notebook file to generate .cnf files for testing/training/validation
|  |--/models
|  |  |--/model_validation          # collection of model architectures and hyperparamter tuning for validation
|  |  |--classifier.py              # final classification model training script
|  |  |--regressor.py               # final regression model training script
|  |--/utils                        # utility functions for generating data
|  |--README.md                     # 
```
### Important Note:
The project submission will not contain the number of `.cnf` files indicated above due to excessive size, instead a small sample will be provided in each directory. 




# CUT
# CUT
# CUT
# CUT
# CUT
## Ideas for analyzing dataset
For each of the multi-variable models implemented below they will consider the following 6 variants of input data:
* Basic Configs (bc)
* Resistant Configs (rc)
* Basic & Resistant Configs (brc)
* Basic Configs + Flipped Lits (bcf)
* Resistant Configs + Flipped Lits (rcf)
* Basic & Resistant Configs + Flipped Lits (brcf)

### Simple linear regression with each input feature

### Basic exploration:
Scale all inputs for these
* KNN
* Multi-variate Linear Regression (Elastic Net Regression)

### Neural approaches:
* Shallow (single & double layer) Neural Net
* Deep Neural Net
* Deep Neural Net w/ Residual Connections
* "Tabular" Transformer
* Graph Neural Network ? 