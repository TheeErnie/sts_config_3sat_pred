import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import optuna
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay

# set up gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# read in (training then testing) data
data = pd.read_csv('../cnf_data/train/train_data.csv')
data['has_solution'] = (data['sol_count'] > 0).astype(int)
y_data = data['has_solution']

test_data = pd.read_csv('../cnf_data/test/test_data.csv')
test_data['has_solution'] = (test_data['sol_count'] > 0).astype(int)
y_test_data = test_data['has_solution']

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
loss_fn = nn.BCEWithLogitsLoss()

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
all_probs = []
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

    # convert to values for use with sklearn metrics
    probs = torch.sigmoid(logits)
    preds = torch.round(probs)

    all_preds.extend(preds.cpu().numpy())
    all_probs.extend(probs.cpu().numpy())
    all_targets.extend(y_batch.cpu().numpy())

# get final average loss
val_loss = val_loss_total / X_test.size(0)

# calculate metrics with sklearn
accuracy = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds, zero_division=0)
recall = recall_score(all_targets, all_preds, zero_division=0)
f1 = f1_score(all_targets, all_preds, zero_division=0)
roc = roc_auc_score(all_targets, all_probs)

print(f'Test Loss: {val_loss:.4f}')
print(f'Accuracy:  {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall:    {recall:.4f}')
print(f'F1:        {f1:.4f}')
print(f'ROC AUC:   {roc:.4f}')

# generate and save confusion matrix
cm = confusion_matrix(all_targets, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Solution', 'Has Solution'])

fig, ax = plt.subplots(figsize=(8,6))
disp.plot(cmap='Blues', values_format='d', ax=ax)

plt.title('Tab Transformer Classifier Conf. Mat.')
plot_file = 'confusion_matrix_ttc.png'
plt.savefig(plot_file)

fig, ax = plt.subplots(figsize=(8,6))
# generate and save roc curve
RocCurveDisplay.from_predictions(all_targets, all_probs, ax=ax)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)

plot_file = 'roc_curve_ttc.png'
plt.savefig(plot_file)