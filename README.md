# TDA_on_MNIST

## MODEL FLAGS
- Only for dataset preparation:
    - FOR MNIST
    ```bash
        MODEL FLAGS = "--data MNIST --ippath None --data_path ../data/datasets_MNIST/"
    ```

    - FOR CMATER
    ```bash
        MODEL FLAGS = "--data CMATER --ippath ../data/CMATERdb3.1.3.3/ --data_path ../data/datasets_CMATER/"
    ```

- Only for topological data extraction (728 features):

    - FOR MNIST
    ```bash
        MODEL FLAGS = "--data MNIST --ippath ../data/datasets_MNIST/ --oppath ../data/datasets_MNIST/tda_features/"
    ```

    - FOR CMATERdb
    ``` bash
        MODEL FLAGS = "--data CMATER --ippath  ../data/datasets_CMATER/ --oppath ../data/datasets_CMATER/tda_features/"
    ```



- Only for Deep Feature Extractions (ResNet18 Model):

    - FOR MNIST
    ``` bash
        MODEL FLAGS = 
    ```

    - FOR CMATER
    ```bash
        MODEL FLAGS = 
    ```
