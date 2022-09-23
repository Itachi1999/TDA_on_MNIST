'''This file contains the code for which measure the classification.
    This code will run for different datasets.
    This code will run for with optimisation and without optimisation.
    Also, we do this with or without concatenating the deep features. 
    This is regulated with the MODEL FLAGS given in the README file given in the repository.'''


import argparse
import codes.utils as utils
import numpy as np

def main():
    parameters = dict_for_classification()
    parser = argparse.ArgumentParser()
    utils.add_dict_to_argparser(parser = parser, default_dict = parameters)
    args = parser.parse_args()

    X_train_tda = np.load(args.data_dir + 'tda_features/X_train_tda.npy', allow_pickle = True)
    X_val_tda = np.load(args.data_dir + 'tda_features/X_val_tda.npy', allow_pickle = True)
    X_test_tda = np.load(args.data_dir + 'tda_features/X_test_tda.npy', allow_pickle = True)



def dict_for_classification():
    return dict(data = "MNIST", data_dir = '../data/datasets_MNIST/', 
        model =  "SVM_multiclass", optimisation = "yes", N = 25, max_iter = 10, concatenate = "yes")


if __name__ == "__main__":
    main()