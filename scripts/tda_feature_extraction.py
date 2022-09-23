from codes import piplines as piplines, utils 
import argparse
import numpy as np




def main():
    paramters = dict_for_feature_extract()
    parser = argparse.ArgumentParser()
    utils.add_dict_to_argparser(parser = parser, default_dict = paramters)
    args = parser.parse_args()

    X_train = np.load(args.ippath + 'train/X_train.npy', allow_pickle=True)
    X_val = np.load(args.ippath + 'validation/X_val.npy', allow_pickle=True)
    X_test = np.load(args.ippath + 'test/X_test.npy', allow_pickle=True)

    # y_train = np.load(args.ippath + 'train/y_train.npy', allow_pickle=True)
    # y_val = np.load(args.ippath + 'validation/y_val.npy', allow_pickle=True)
    # y_test = np.load(args.ippath + 'test/y_test.npy', allow_pickle=True)
    tda_union = piplines.paper_pipeline()

    print('Starting the tda process for X_train\n')
    X_train_tda = tda_union.fit_transform(X_train)
    print(f'Shape of X_train_tda: {X_train_tda.shape}')
    print('Ending tda process for X_train\n')
    print('Saving X_train_tda\n')
    np.save(args.oppath + 'X_train_tda.npy', X_train_tda)
    print('Saved\n')

    print('Starting the tda process for X_val\n')
    X_val_tda = tda_union.fit_transform(X_val)
    print(f'Shape of X_val_tda: {X_val_tda.shape}')
    print('Ending tda process for X_val\n')
    print('Saving X_val_tda\n')
    np.save(args.oppath + 'X_val_tda.npy', X_val_tda)
    print('Saved\n')


    print('Starting the tda process for X_test\n')
    X_test_tda = tda_union.fit_transform(X_test)
    print(f'Shape of X_test_tda: {X_test_tda.shape}')
    print('Ending tda process for X_test\n')
    print('Saving X_test_tda\n')
    np.save(args.oppath + 'X_test_tda.npy', X_test_tda)
    print('Saved\n')





def dict_for_feature_extract():
    return dict(data = "MNIST", ippath = "../data/datasets_MNIST/", oppath = "data/datasets_MNIST/tda_features/")



if __name__ == "__main__":
    main()


