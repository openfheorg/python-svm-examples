# OpenFHE - SVM
OpenFHE-Python-based Examples of Encrypted Support Vector Machine Inference based on [this C++ example](https://github.com/caesaretos/svm-fhe/tree/main).

## Running

First, install `openfhe-python` following the instructions of [library's repository](https://github.com/openfheorg/openfhe-python/).

Then, you can run `python model_training.py` to train and save the model weights, and then call the encrypted model files for inference. All necessary data is inside `data/` directory, but if you want to reproduce the data generation run the `get_data.py` file.

