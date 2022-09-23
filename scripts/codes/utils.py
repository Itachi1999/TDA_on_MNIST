from sklearn.metrics import classification_report, confusion_matrix
import os
import cv2
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

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



def dataset_preparation_CMATER(path):
        X = []
        y = []
        list = os.listdir(path)
        for i, l in enumerate(list):
                p = os.path.join(path, l)
                print(p)
                for str in os.listdir(p):
                        #print(str)
                        d = os.path.join(p, str)
                        if('.bmp' in str):
                                img_gray=cv2.imread(d, 0)
                                y.append(l)
                                img_res = cv2.resize(img_gray, (28, 28), interpolation = cv2.INTER_NEAREST)
                                X.append(img_res)
                        else:
                                for s in os.listdir(d):
                                        #print(s)
                                        #print('except')
                                        img_gray = cv2.imread(os.path.join(d, s), 0)
                                        y.append(l + '_' + str)
                                        img_res = cv2.resize(img_gray, (28, 28), interpolation = cv2.INTER_CUBIC)
                                        X.append(img_res)
        X = np.asarray(X)
        y = np.asarray(y)
        #print(X[0])
        return X, y  


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

