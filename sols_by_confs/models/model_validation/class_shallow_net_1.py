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

# simple single layer model
class OneLayerNN(nn.Module):
  def __init__(self, input_dim, hidden_dim, activation):
    super().__init__()
    self.hidden = nn.Linear(input_dim, hidden_dim)
    self.output = nn.Linear(hidden_dim,1)
    self.activation = activation

    # TODO
    with torch.no_grad():
      self.output.bias.fill_(-2.7)

  def forward(self, x):
    x = self.activation(self.hidden(x))
    return self.output(x)

# define objective function for tuning
def objective(trial):
  NUM_EPOCHS = 10
  # params to tune
  hidden_size = trial.suggest_int('hidden_size', 32, 512, step=32)
  lr = trial.suggest_float('lr', 1e-5,1e-2)

  activation_name = trial.suggest_categorical(
    'activation', ['ReLU', 'GELU', 'ELU', 'SiLU']
  )

  activations = { 
    'ReLU': nn.ReLU(),
    'GELU': nn.GELU(),
    'ELU': nn.ELU(),
    'SiLU': nn.SiLU()
  }
  activation_fn = activations[activation_name]

  # model setup
  model = OneLayerNN(input_dim=input_dim, hidden_dim=hidden_size, activation=activation_fn).to(device)
  criterion = nn.BCEWithLogitsLoss()#pos_weight=weight)
  optimizer = optim.Adam(model.parameters(), lr=lr)

  # training loop
  for epoch in range(NUM_EPOCHS):
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