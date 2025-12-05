# Documentation for sklearn LinearRegression:
#   https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

from sklearn.linear_model import LinearRegression
import pandas as pd

# import test data
df = pd.read_csv('../cnf_data/test/test_data.csv')
y = df[['sol_count']]

# set up each variable for regression
X_lit_flip = df[['neg_lits_count']]
X_pasch = df[['pasch_count']]
X_mitre = df[['mitre_count']]
X_fanol = df[['fano_line_count']]
X_grid  = df[['grid_count']]
X_prism = df[['prism_count']]
X_hexag = df[['hexagon_count']]
X_crown = df[['crown_count']]
X_r_pasch = df[['pasch_count_r']]
X_r_mitre = df[['mitre_count_r']]
X_r_fanol = df[['fano_line_count_r']]
X_r_grid  = df[['grid_count_r']]
X_r_prism = df[['prism_count_r']]
X_r_hexag = df[['hexagon_count_r']]
X_r_crown = df[['crown_count_r']]

# fit each variable to regression line
reg_lit_flip = LinearRegression().fit(X_lit_flip, y)
reg_pasch = LinearRegression().fit(X_pasch, y)
reg_mitre = LinearRegression().fit(X_mitre, y)
reg_fanol = LinearRegression().fit(X_fanol, y)
reg_grid  = LinearRegression().fit(X_grid, y)
reg_prism = LinearRegression().fit(X_prism, y)
reg_hexag = LinearRegression().fit(X_hexag, y)
reg_crown = LinearRegression().fit(X_crown, y)
reg_r_pasch = LinearRegression().fit(X_r_pasch, y)
reg_r_mitre = LinearRegression().fit(X_r_mitre, y)
reg_r_fanol = LinearRegression().fit(X_r_fanol, y)
reg_r_grid  = LinearRegression().fit(X_r_grid, y)
reg_r_prism = LinearRegression().fit(X_r_prism, y)
reg_r_hexag = LinearRegression().fit(X_r_hexag, y)
reg_r_crown = LinearRegression().fit(X_r_crown, y)

# store each regression line's score
score_lit_flip = reg_lit_flip.score(X_lit_flip, y)
score_pasch = reg_pasch.score(X_pasch, y)
score_mitre = reg_mitre.score(X_mitre, y)
score_fanol = reg_fanol.score(X_fanol, y)
score_grid  = reg_grid.score(X_grid, y)
score_prism = reg_prism.score(X_prism, y)
score_hexag = reg_hexag.score(X_hexag, y)
score_crown = reg_crown.score(X_crown, y)
score_r_pasch = reg_r_pasch.score(X_r_pasch, y)
score_r_mitre = reg_r_mitre.score(X_r_mitre, y)
score_r_fanol = reg_r_fanol.score(X_r_fanol, y)
score_r_grid  = reg_r_grid.score(X_r_grid, y)
score_r_prism = reg_r_prism.score(X_r_prism, y)
score_r_hexag = reg_r_hexag.score(X_r_hexag, y)
score_r_crown = reg_r_crown.score(X_r_crown, y)

# print out scores
print(f'====== Simple, Single Linear Regression R^2 Scores ======')
print(f'Pasch: {score_pasch:.6f}, resistant: {score_r_pasch:.6f}')
print(f'Mitre: {score_mitre:.6f}, resistant: {score_r_mitre:.6f}')
print(f'FanoL: {score_fanol:.6f}, resistant: {score_r_fanol:.6f}')
print(f'Grid:  {score_grid:.6f}, resistant: {score_r_grid:.6f}')
print(f'Hexag: {score_hexag:.6f}, resistant: {score_r_hexag:.6f}')
print(f'Crown: {score_crown:.6f}, resistant: {score_r_crown:.6f}')
print(f'Lit Flip: {score_lit_flip:.6f}')