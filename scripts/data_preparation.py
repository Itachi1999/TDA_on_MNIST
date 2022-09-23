import argparse
from codes import utils

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import numpy as np



def main():
    paramters = dict_for_data()
    parser = argparse.ArgumentParser()
    utils.add_dict_to_argparser(parser = parser, default_dict = paramters)
    args = parser.parse_args()

    print(f"Starting to prepare data for {args.data} ...")
    if(args.data == "MNIST"):
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        np.save(args.data_path + 'raw_data/X_raw.npy', X)
        np.save(args.data_path + 'raw_data/y_raw.npy', y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 ,
                                    stratify=y, random_state=666)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 666)


        X_train = X_train.reshape(X_train.shape[0], 28, 28)
        X_val = X_val.reshape(X_val.shape[0], 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 28, 28)
    
    else:
        X_train, y_train = utils.dataset_preparation_CMATER(args.ippath + 'IsolatedTrain/')
        X_test, y_test = utils.dataset_preparation_CMATER(args.ippath + 'IsolatedTest/')

        train_size = X_train.shape[0]
        test_size = X_test.shape[0]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size = (train_size - test_size),
                                                        test_size = test_size, random_state = 666)
    
    print("Finished processing data ...")
    np.save(args.data_path + 'train/X_train.npy', X_train)
    np.save(args.data_path + 'test/X_test.npy', X_test)
    np.save(args.data_path + 'validation/X_val.npy', X_val)
    np.save(args.data_path + 'validation/y_val.npy', y_val)
    np.save(args.data_path + 'train/y_train.npy', y_train)
    np.save(args.data_path + 'test/y_test.npy', y_test)

    print("Details of the data: \n")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val Shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")



def dict_for_data():
    return dict(data = "MNIST", ippath = None, data_path = "../data/datasets_MNIST/")


if __name__ == "__main__":
    main()