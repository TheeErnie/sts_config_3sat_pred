import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import optuna
import sys

# set up gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# read in data
data = pd.read_csv('../cnf_data/val/val_data.csv')
data['has_solution'] = (data['sol_count'] > 0).astype(int)
y_data = data['has_solution']

# alter data based on input arguments
if len(sys.argv) > 1:
  match sys.argv[1]:
    case 'rc':
      X_data = data[['pasch_count_r', 'mitre_count_r', 'fano_line_count_r', 'grid_count_r', 'prism_count_r', 'hexagon_count_r', 'crown_count_r']]
      input_dim = X_data.shape[1]
    case 'brc':
      X_data = data[['pasch_count', 'mitre_count', 'fano_line_count', 'grid_count', 'prism_count', 'hexagon_count', 'crown_count', 'pasch_count_r', 'mitre_count_r', 'fano_line_count_r', 'grid_count_r', 'prism_count_r', 'hexagon_count_r', 'crown_count_r']]
      input_dim = X_data.shape[1]
    case 'bcf':
      X_data = data[['neg_lits_count', 'pasch_count', 'mitre_count', 'fano_line_count', 'grid_count', 'prism_count', 'hexagon_count', 'crown_count']]
      input_dim = X_data.shape[1]
    case 'rcf':
      X_data = data[['neg_lits_count', 'pasch_count_r', 'mitre_count_r', 'fano_line_count_r', 'grid_count_r', 'prism_count_r', 'hexagon_count_r', 'crown_count_r']]
      input_dim = X_data.shape[1]
    case 'brcf':
      X_data = data[['neg_lits_count', 'pasch_count', 'mitre_count', 'fano_line_count', 'grid_count', 'prism_count', 'hexagon_count', 'crown_count', 'pasch_count_r', 'mitre_count_r', 'fano_line_count_r', 'grid_count_r', 'prism_count_r', 'hexagon_count_r', 'crown_count_r']]
      input_dim = X_data.shape[1]
    case _:
      X_data = data[['pasch_count','mitre_count','fano_line_count','grid_count','prism_count','hexagon_count','crown_count']]
      input_dim = X_data.shape[1]
else:
  X_data = data[['pasch_count','mitre_count','fano_line_count','grid_count','prism_count','hexagon_count','crown_count']]
  input_dim = X_data.shape[1]

# cast into tensors and send to (hopefully) gpu
X = torch.tensor(X_data.values, dtype=torch.float32)
y = torch.tensor(y_data.values, dtype=torch.float32).view(-1,1)
X = X.to(device)
y = y.to(device)

# simple double layer model
class TwoLayerNN(nn.Module):
  def __init__(self, input_dim, hidden_dim_0, hidden_dim_1, activation_0, activation_1):
    super().__init__()
    self.hidden_0 = nn.Linear(input_dim, hidden_dim_0)
    self.hidden_1 = nn.Linear(hidden_dim_0, hidden_dim_1)
    self.output = nn.Linear(hidden_dim_1, 1)
    self.activation_0 = activation_0
    self.activation_1 = activation_1

  def forward(self, x):
    x = self.activation_0(self.hidden_0(x))
    x = self.activation_1(self.hidden_1(x))
    return self.output(x)

# define objective function for tuning
def objective(trial):
  # params to tune
  hidden_size_0 = trial.suggest_int('hidden_size_0', 32, 512, step=32)
  hidden_size_1 = trial.suggest_int('hidden_size_1', 32, 512, step=32)
  lr = trial.suggest_float('lr', 1e-5,1e-2)
  num_epochs = trial.suggest_int('num_epochs', 5, 50)

  activation_name_0 = trial.suggest_categorical(
    'activation_0', ['ReLU', 'GELU', 'ELU', 'SiLU']
  )
  activation_name_1 = trial.suggest_categorical(
    'activation_1', ['ReLU', 'GELU', 'ELU', 'SiLU']
  )

  activations = {
    'ReLU': nn.ReLU(),
    'GELU': nn.GELU(),
    'ELU': nn.ELU(),
    'SiLU': nn.SiLU()
  }
  activation_fn_0 = activations[activation_name_0]
  activation_fn_1 = activations[activation_name_1]

  # model setup
  model = TwoLayerNN(input_dim=input_dim, hidden_dim_0=hidden_size_0, hidden_dim_1=hidden_size_1, activation_0=activation_fn_0, activation_1=activation_fn_1).to(device)
  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)

  # training loop
  for epoch in range(num_epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return loss.item()

# run study
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print('Best Trial:')
print(study.best_trial.params)