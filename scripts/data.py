#from ensurepip import version
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# #Saving the raw X and y into CSV's
# X.to_csv('../data/X_raw.csv', index = False)
# y.to_csv('../data/y_raw.csv', index = False)

# #Printing the shape of raw data files
# print(f"Shape of X: {X.shape}, Shape pf y; {y.shape}")

# #Trying the trainig with 6000 images
train_size, test_size = 6000, 1000

X = pd.read_csv('../data/X_raw.csv', index_col= False)
y = pd.read_csv('../data/y_raw.csv', index_col= False)


#Reshape to (n_samples, n_pixels_x, n_pixels_y)
#X = X.reshape((-1, 28, 28))
X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_size, test_size=test_size, stratify=y, random_state=666
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

pd.DataFrame(X_train).to_csv('../data/X_train.csv')
pd.DataFrame(y_train).to_csv('../data/y_train.csv')
pd.DataFrame(X_test).to_csv('../data/X_test.csv' )
pd.DataFrame(y_test).to_csv('../data/y_test.csv')