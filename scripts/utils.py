from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def confusion_matrix_creation(filename, y_test, y_pred_test, title, class_names):
    matrix = confusion_matrix(y_test, y_pred_test)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    # Build the plot
    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Blues, linewidths=0.2)

    # Add labels to the plot
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title)
    plt.savefig('../results_log/confusion_matrices/confusion_matrix_' + filename + '.png')


def readCSV(filepath):
        df = pd.read_csv(filepath)
        df = df.iloc[::, 1::]
        data = df.to_numpy()

        return data


def saveCSV(file, filepath, fileName):
        df = pd.DataFrame(file, index= None)
        df.to_csv(filepath + fileName + '.csv')
        return None

def writeClassificationReport(y_test, y_pred_test, filename, filepath = '../results_log/classification_reports/'):
        with open(filepath + filename + '.txt', 'w') as writefile:
                writefile.write(classification_report(y_test, y_pred_test))      
