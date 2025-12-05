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

# residual block for model
class ResidualBlock(nn.Module):
  def __init__(self, in_dim, out_dim, activation, dropout_rate=0.0):
    super().__init__()
    self.linear = nn.Linear(in_dim, out_dim)
    self.activation = activation
    self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
    self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

  def forward(self, x):
    out = self.linear(x)
    out = self.activation(out)
    out = self.dropout(out)
    out = out + self.shortcut(x)
    return out

# deep feedforward neural net
class DeepNN(nn.Module):
  def __init__(self, input_dim, hidden_dims, activations, dropout_rate=0.0):
    super().__init__()
    # make sure the inputs are valid
    assert len(hidden_dims) == len(activations), \
      'Each hidden layer must have a corresponding activation.'

    layers = []
    prev_dim = input_dim

    # loop and assign layers
    for hdim, act in zip(hidden_dims, activations):
      layers.append(ResidualBlock(prev_dim, hdim, act, dropout_rate))
      prev_dim = hdim

    # set up output layer for regression
    layers.append(nn.Linear(prev_dim, 1)) 
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)

# define objective function for tuning
def objective(trial):
  # params to tune
  n_layers = trial.suggest_int('n_layers', 2, 7)
  hidden_dims = [trial.suggest_int(f'hidden_dim_{i}', 32, 256, log=True) for i in range(n_layers)]
  activation_names = [trial.suggest_categorical(f'activation_{i}', ['relu', 'tanh', 'elu', 'gelu', 'silu']) for i in range(n_layers)]
  activation_map = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'elu': nn.ELU(),
    'gelu': nn.GELU(),
    'silu': nn.SiLU(),
  }
  activations = [activation_map[name] for name in activation_names]
  dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
  lr=trial.suggest_float('lr', 1e-5, 1e-2, log=True)

  # set model
  model = DeepNN(input_dim=input_dim, hidden_dims=hidden_dims, activations=activations, dropout_rate=dropout_rate).to(device)

  # optimizer and loss
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  loss_fn = nn.BCEWithLogitsLoss()

  # training loop
  epochs = 25
  for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

  model.eval()
  with torch.no_grad():
    val_pred = model(X)
    val_loss = loss_fn(val_pred, y).item()

  return val_loss

# run study
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print('Best Trial:')
print(study.best_trial.params)