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
y_data = data['sol_count']

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

class TabTransformer(nn.Module):
  def __init__(self, num_features, vocab_sizes=None, emb_dim=32, num_layers=2, num_heads=4, hidden_dim=64, dropout=0.1):
    super().__init__()
    
    # linear embedding for continuous int values
    self.input_proj = nn.Linear(1, emb_dim)

    encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    # output head
    self.head = nn.Sequential(
      nn.Flatten(),
      nn.Linear(num_features*emb_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim,1)
    )

  def forward(self, x):
    x = x.unsqueeze(-1)
    x = self.input_proj(x)
    x = self.transformer(x)
    return self.head(x)

# define objective function for tuning
def objective(trial):
  # params to tune
  num_heads = trial.suggest_int('num_heads', 2, 8, step=2)
  emb_dim = trial.suggest_int('emb_dim', num_heads, 32, step=num_heads)
  num_layers = trial.suggest_int('num_layers', 2, 5)
  hidden_dim = trial.suggest_int('hidden_dim', 32, 64, step=32)
  dropout = trial.suggest_float('dropout', 0.0, 0.5)
  lr=trial.suggest_float('lr', 1e-5, 1e-2, log=True)

  # set model
  model = TabTransformer(num_features=X.shape[1], emb_dim=emb_dim, num_layers=num_layers, num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout).to(device)

  # optimizer and loss
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  loss_fn = nn.HuberLoss(delta=1.0)

  # training loop
  batch_size = 1024
  epochs = 5 #25
  for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X.size(0))
    for i in range(0, X.size(0), batch_size):
      indices = permutation[i:i + batch_size]
      X_batch = X[indices]
      y_batch = y[indices]

      optimizer.zero_grad()
      y_pred = model(X_batch)
      loss = loss_fn(y_pred, y_batch)
      loss.backward()
      optimizer.step()

  model.eval()
  val_loss_total = 0.0
  with torch.no_grad():
    for i in range(0, X.size(0), batch_size):
      indices = permutation[i:i + batch_size]
      X_batch = X[indices]
      y_batch = y[indices]
      val_pred = model(X_batch)
      val_loss_total += loss_fn(val_pred,y_batch).item() * X_batch.size(0)
    val_loss = val_loss_total / X.size(0)

  return val_loss

# run study
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print('Best Trial:')
print(study.best_trial.params)