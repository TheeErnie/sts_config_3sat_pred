import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import optuna
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# set up gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# read in (training then testing) data
data = pd.read_csv('../cnf_data/train/train_data.csv')
y_data = data['sol_count']

test_data = pd.read_csv('../cnf_data/test/test_data.csv')
y_test_data = test_data['sol_count']

# using resistant counts only (determined in model validation)
# training data
X_data = data[['pasch_count_r', 'mitre_count_r', 'fano_line_count_r', 'grid_count_r', 'prism_count_r', 'hexagon_count_r', 'crown_count_r']]
# testing data
X_test_data = test_data[['pasch_count_r', 'mitre_count_r', 'fano_line_count_r', 'grid_count_r', 'prism_count_r', 'hexagon_count_r', 'crown_count_r']]
input_dim = X_data.shape[1]

# cast into tensors and send to (hopefully) gpu
# train data
X = torch.tensor(X_data.values, dtype=torch.float32).to(device)
y = torch.tensor(y_data.values, dtype=torch.float32).view(-1,1).to(device)

# test data
X_test = torch.tensor(X_test_data.values, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test_data.values, dtype=torch.float32).view(-1,1).to(device)

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

# manually setting hyperparameters based on model validation / param tuning
num_heads = 6
emb_dim = 24
num_layers = 2
hidden_dim = 32
dropout = 0.002
lr = 0.000000444435

model = TabTransformer(num_features=X.shape[1], emb_dim=emb_dim, num_layers=num_layers, num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.HuberLoss(delta=1.0)


EPOCHS = 10
batch_size = 1024

for epoch in range(EPOCHS):
  model.train()
  permutation = torch.randperm(X.size(0))

  with tqdm(range(0, X.size(0), batch_size), desc=f'Epoch {epoch+1}/{EPOCHS}') as pbar:
    for i in pbar:
      indices = permutation[i:i+batch_size]
      X_batch = X[indices]
      y_batch = y[indices]

      optimizer.zero_grad()
      y_pred = model(X_batch)
      loss = loss_fn(y_pred, y_batch)
      loss.backward()
      optimizer.step()

      pbar.set_postfix(loss=f'{loss.item():.4f}')

# evaluate trained model
model.eval()

# variables to hold results from all batches
all_preds = []
all_targets = []
val_loss_total = 0.0

with torch.no_grad():
  for i in range(0, X_test.size(0), batch_size):
    # slice test data to grab a batch
    X_batch = X_test[i:i+batch_size]
    y_batch = y_test[i:i+batch_size]

    # forward pass through the model
    logits = model(X_batch)

    # calculate loss 
    val_loss_total += loss_fn(logits, y_batch).item() * X_batch.size(0)

    all_preds.extend(logits.cpu().numpy())
    all_targets.extend(y_batch.cpu().numpy())

# get final average loss
val_loss = val_loss_total / X_test.size(0)

# calculate metrics with sklearn
mse = mean_squared_error(all_targets, all_preds)
mae = mean_absolute_error(all_targets, all_preds)
r2 = r2_score(all_targets, all_preds)

print(f'Test Loss: {val_loss:.4f}')
print(f'MSE:       {mse:.4f}')
print(f'MAE:       {mae:.4f}')
print(f'R2:        {r2:.4f}')


# Evaluation of random instances

print('\n' + '='*34)
print('  SAMPLED TRAINING INSPECTION')
print('='*34)

VARIABLE = 10  

model.eval()
rand_indices = torch.randint(0, X.size(0), (VARIABLE,))

x_sample = X[rand_indices]
y_sample = y[rand_indices]

with torch.no_grad():
    pred_sample = model(x_sample)

x_disp = x_sample.cpu().numpy()
y_disp = y_sample.cpu().numpy()
p_disp = pred_sample.cpu().numpy()

print(f"{'INPUT FEATURES (7 Counts)':<24} | {'TRUE':<8} | {'PRED':<8}")
print("-" * 36)

for i in range(VARIABLE):
    inputs_str = ", ".join([f"{val:.0f}" for val in x_disp[i]])
    
    true_val = y_disp[i].item()
    pred_val = p_disp[i].item()
    
    print(f"[{inputs_str:<22}] | {true_val:<8.0f} | {pred_val:<8.2f}")