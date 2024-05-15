#%%
import numpy as np
import pandas as pd

n_data_points = int(1e5)

df = pd.DataFrame()

x_vars = 5

for i in range(x_vars):
    df[f"x{i}"] = np.random.normal(size=n_data_points)

df["y"] = np.sin(df['x1'] * df['x2']) + np.abs(df['x3'])**(df['x4'])
df['Tracking'] = np.arange(n_data_points)
#%%
df.to_csv("fake_data.csv")
# %%
