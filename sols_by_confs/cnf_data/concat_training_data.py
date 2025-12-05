import pandas as pd

df0 = pd.read_csv('train/train_sub0_data.csv')
df1 = pd.read_csv('train/train_sub1_data.csv')
df2 = pd.read_csv('train/train_sub2_data.csv')
df3 = pd.read_csv('train/train_sub3_data.csv')
df4 = pd.read_csv('train/train_sub4_data.csv')

df_out = pd.concat([df0,df1,df2,df3,df4], ignore_index=True)

df_out.to_csv('train/train_data.csv')
