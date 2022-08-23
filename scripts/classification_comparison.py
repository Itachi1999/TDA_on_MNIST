from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import utils


class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

X_test_tda = utils.readCSV('../data/tda_features/X_test_tda.csv')
X_train_tda = utils.readCSV('../data/tda_features/X_train_tda.csv')
X_test_reduced_PCA = utils.readCSV('../data/reduced_tda_features/PCA_28/X_test_reduced_PCA.csv')
X_train_reduced_PCA = utils.readCSV('../data/reduced_tda_features/PCA_28/X_train_reduced_PCA.csv')
y_train = utils.readCSV('../data/y_train.csv')
y_test = utils.readCSV('../data/y_test.csv')

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)
#print(f'{X_test_reduced_PCA.shape}')


#Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train_tda, y_train)

rf_reduced_PCA = RandomForestClassifier()
rf_reduced_PCA.fit(X_train_reduced_PCA, y_train)

y_pred_test = rf.predict(X_test_tda)
y_pred_test_reduced_PCA = rf_reduced_PCA.predict(X_test_reduced_PCA)


utils.writeClassificationReport(y_test=y_test, y_pred_test=y_pred_test, filename='RF')
utils.writeClassificationReport(y_test=y_test, y_pred_test=y_pred_test_reduced_PCA, filename='RF_PCA28')

utils.confusion_matrix_creation('RF', y_test, y_pred_test, 'Random Forest Classifier', class_names)
utils.confusion_matrix_creation('RF_PCA28', y_test, y_pred_test_reduced_PCA, 'Random Forest with PCA 28', class_names)





