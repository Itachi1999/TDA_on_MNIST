from operator import index
import pandas as pd 
import numpy as np
from gtda.plotting import plot_heatmap

X_train = pd.read_csv('../data/X_train.csv', index_col=False)
y_train = pd.read_csv('../data/y_train.csv', index_col=False)


print(f'Shape of X_train: {X_train.shape}')
print(X_train.loc[1])

X_train = X_train.iloc[::, 1::]
y_train = y_train["0"]
num_x = X_train.shape[0]
num_y = y_train.shape[0]
X_train = X_train.to_numpy().reshape((num_x, 28, 28))
y_train = y_train.to_numpy().reshape((num_y, ))

print(f'Shape of X-train: {X_train.shape}, Shape of y: {y_train.shape}')



im8_idx = np.flatnonzero(y_train == 8)[0]
img8 = X_train[im8_idx].reshape(28, 28)
plot_heatmap(img8)