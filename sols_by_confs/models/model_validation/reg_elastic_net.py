# Documentation for sklearn's ElasticNet regression
#   https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet
# Documentation for optuna

import pandas as pd
import numpy as np
import optuna
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# read in data
df_test = pd.read_csv('../cnf_data/test/test_data.csv')
df_validate = pd.read_csv('../cnf_data/val/val_data.csv')
y_val = df_validate[['sol_count']]
y_test = df_test[['sol_count']]

# set established input list variants
X_val_bc = df_validate[['pasch_count', 'mitre_count', 'fano_line_count', 'grid_count', 'prism_count', 'hexagon_count', 'crown_count']]
X_val_rc = df_validate[['pasch_count_r', 'mitre_count_r', 'fano_line_count_r', 'grid_count_r', 'prism_count_r', 'hexagon_count_r', 'crown_count_r']]
X_val_brc = df_validate[['pasch_count', 'mitre_count', 'fano_line_count', 'grid_count', 'prism_count', 'hexagon_count', 'crown_count', 'pasch_count_r', 'mitre_count_r', 'fano_line_count_r', 'grid_count_r', 'prism_count_r', 'hexagon_count_r', 'crown_count_r']]
X_val_bcf = df_validate[['neg_lits_count', 'pasch_count', 'mitre_count', 'fano_line_count', 'grid_count', 'prism_count', 'hexagon_count', 'crown_count']]
X_val_rcf = df_validate[['neg_lits_count', 'pasch_count_r', 'mitre_count_r', 'fano_line_count_r', 'grid_count_r', 'prism_count_r', 'hexagon_count_r', 'crown_count_r']]
X_val_brcf = df_validate[['neg_lits_count', 'pasch_count', 'mitre_count', 'fano_line_count', 'grid_count', 'prism_count', 'hexagon_count', 'crown_count', 'pasch_count_r', 'mitre_count_r', 'fano_line_count_r', 'grid_count_r', 'prism_count_r', 'hexagon_count_r', 'crown_count_r']]
X_test_bc = df_test[['pasch_count', 'mitre_count', 'fano_line_count', 'grid_count', 'prism_count', 'hexagon_count', 'crown_count']]
X_test_rc = df_test[['pasch_count_r', 'mitre_count_r', 'fano_line_count_r', 'grid_count_r', 'prism_count_r', 'hexagon_count_r', 'crown_count_r']]
X_test_brc = df_test[['pasch_count', 'mitre_count', 'fano_line_count', 'grid_count', 'prism_count', 'hexagon_count', 'crown_count', 'pasch_count_r', 'mitre_count_r', 'fano_line_count_r', 'grid_count_r', 'prism_count_r', 'hexagon_count_r', 'crown_count_r']]
X_test_bcf = df_test[['neg_lits_count', 'pasch_count', 'mitre_count', 'fano_line_count', 'grid_count', 'prism_count', 'hexagon_count', 'crown_count']]
X_test_rcf = df_test[['neg_lits_count', 'pasch_count_r', 'mitre_count_r', 'fano_line_count_r', 'grid_count_r', 'prism_count_r', 'hexagon_count_r', 'crown_count_r']]
X_test_brcf = df_test[['neg_lits_count', 'pasch_count', 'mitre_count', 'fano_line_count', 'grid_count', 'prism_count', 'hexagon_count', 'crown_count', 'pasch_count_r', 'mitre_count_r', 'fano_line_count_r', 'grid_count_r', 'prism_count_r', 'hexagon_count_r', 'crown_count_r']]

# scale data (not necessary but may be helpful)
scaler = StandardScaler()
X_val_scaled_bc = scaler.fit_transform(X_val_bc)
X_val_scaled_rc = scaler.fit_transform(X_val_rc)
X_val_scaled_brc = scaler.fit_transform(X_val_brc)
X_val_scaled_bcf = scaler.fit_transform(X_val_bcf)
X_val_scaled_rcf = scaler.fit_transform(X_val_rcf)
X_val_scaled_brcf = scaler.fit_transform(X_val_brcf)
X_test_scaled_bc = scaler.fit_transform(X_test_bc)
X_test_scaled_rc = scaler.fit_transform(X_test_rc)
X_test_scaled_brc = scaler.fit_transform(X_test_brc)
X_test_scaled_bcf = scaler.fit_transform(X_test_bcf)
X_test_scaled_rcf = scaler.fit_transform(X_test_rcf)
X_test_scaled_brcf = scaler.fit_transform(X_test_brcf)

# run bayesian optimization on each model to fit hyper-parameters (alpha and l1_ratio)
def objective(trial, X, y): # objective function for tuning
    alpha = trial.suggest_float('alpha', 1e-4, 1e2)
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(X,y)
    y_pred = model.predict(X)
    loss = mean_absolute_error(y, y_pred)
    return loss

optuna.logging.set_verbosity(optuna.logging.ERROR)
study_bc = optuna.create_study(direction='minimize')
study_bc.optimize(lambda trial: objective(trial, X_val_scaled_bc, y_val), n_trials=50, show_progress_bar=False)
study_rc = optuna.create_study(direction='minimize')
study_rc.optimize(lambda trial: objective(trial, X_val_scaled_rc, y_val), n_trials=50, show_progress_bar=False)
study_brc = optuna.create_study(direction='minimize')
study_brc.optimize(lambda trial: objective(trial, X_val_scaled_brc, y_val), n_trials=50, show_progress_bar=False)
study_bcf = optuna.create_study(direction='minimize')
study_bcf.optimize(lambda trial: objective(trial, X_val_scaled_bcf, y_val), n_trials=50, show_progress_bar=False)
study_rcf = optuna.create_study(direction='minimize')
study_rcf.optimize(lambda trial: objective(trial, X_val_scaled_rcf, y_val), n_trials=50, show_progress_bar=False)
study_brcf = optuna.create_study(direction='minimize')
study_brcf.optimize(lambda trial: objective(trial, X_val_scaled_brcf, y_val), n_trials=50, show_progress_bar=False)

# fit EN models with hyperparameters from above
enr_bc = ElasticNet(**study_bc.best_params)
enr_bc.fit(X_test_scaled_bc, y_test)
enr_rc = ElasticNet(**study_rc.best_params)
enr_rc.fit(X_test_scaled_rc, y_test)
enr_brc = ElasticNet(**study_brc.best_params)
enr_brc.fit(X_test_scaled_brc, y_test)
enr_bcf = ElasticNet(**study_bcf.best_params)
enr_bcf.fit(X_test_scaled_bcf, y_test)
enr_rcf = ElasticNet(**study_rcf.best_params)
enr_rcf.fit(X_test_scaled_rcf, y_test)
enr_brcf = ElasticNet(**study_brcf.best_params)
enr_brcf.fit(X_test_scaled_brcf, y_test)

# get model scores (R^2) and print results
r2_bc = enr_bc.score(X_test_scaled_bc, y_test)
r2_rc = enr_rc.score(X_test_scaled_rc, y_test)
r2_brc = enr_brc.score(X_test_scaled_brc, y_test)
r2_bcf = enr_bcf.score(X_test_scaled_bcf, y_test)
r2_rcf = enr_rcf.score(X_test_scaled_rcf, y_test)
r2_brcf = enr_brcf.score(X_test_scaled_brcf, y_test)

print('====== ENR R^2 Scores for Varried Input Combinations ======')
print(f'Basic Configs:                 {r2_bc:.6f}')
print(f'Resistant Configs:             {r2_rc:.6f}')
print(f'Basic & Resistant Configs:     {r2_brc:.6f}')
print(f'Basic + Lit Flips:             {r2_bcf:.6f}')
print(f'Resistant + Lit Flips:         {r2_rcf:.6f}')
print(f'Basic & Resistant + Lit Flips: {r2_brcf:.6f}')