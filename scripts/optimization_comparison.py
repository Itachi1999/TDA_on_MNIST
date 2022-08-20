from sklearn.decomposition import PCA
import pandas as pd

#Dataset load
X_test_tda = pd.read_csv('../data/tda_features/X_test_tda.csv', index_col=None)
X_train_tda = pd.read_csv('../data/tda_features/X_train_tda.csv', index_col=None)


X_test_tda = X_test_tda.iloc[::, 1::]
X_train_tda = X_train_tda.iloc[::, 1::]

X_test_tda = X_test_tda.to_numpy()
X_train_tda = X_train_tda.to_numpy()

print(f'The shape of X_test_tda: {X_test_tda.shape}, Shape of X_train_tda: {X_train_tda.shape}')

#Principal Componenet Analysis
pca_model = PCA(n_components= 28)

X_train_reduced_PCA = pca_model.fit_transform(X_train_tda)
X_test_reduced_PCA = pca_model.fit_transform(X_test_tda)

print(f'The shape of X_test_reduced: {X_test_reduced_PCA.shape}, Shape of X_train_reduced: {X_train_reduced_PCA.shape}')

X_train_reduced_PCA.tofile('../data/reduced_tda_features/PCA_28/X_train_reduced_PCA.csv', sep = ',')
X_test_reduced_PCA.tofile('../data/reduced_tda_features/PCA_28/X_test_reduced_PCA.csv', sep = ',')








