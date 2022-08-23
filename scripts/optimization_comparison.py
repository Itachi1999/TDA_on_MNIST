from sklearn.decomposition import PCA
import pandas as pd
import utils

#Dataset load
X_test_tda = utils.readCSV('../data/tda_features/X_test_tda.csv')
X_train_tda = utils.readCSV('../data/tda_features/X_train_tda.csv')

print(f'The shape of X_test_tda: {X_test_tda.shape}, Shape of X_train_tda: {X_train_tda.shape}')

#Principal Componenet Analysis
pca_model = PCA(n_components= 28)

X_train_reduced_PCA = pca_model.fit_transform(X_train_tda)
X_test_reduced_PCA = pca_model.fit_transform(X_test_tda)

print(f'The shape of X_test_reduced: {X_test_reduced_PCA.shape}, Shape of X_train_reduced: {X_train_reduced_PCA.shape}')

utils.saveCSV(X_test_reduced_PCA, '../data/reduced_tda_features/PCA_28/', 'X_test_reduced_PCA')
utils.saveCSV(X_train_reduced_PCA, '../data/reduced_tda_features/PCA_28/', 'X_train_reduced_PCA')








