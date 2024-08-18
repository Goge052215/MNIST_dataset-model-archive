# MNIST-data-train

## Preview
This repository records various models for the MNIST dataset training and testing. So far, we achieved an accuracy of 99.55% and is currently tesing for other base models
In thie repo we added the base code for training for models like `Neural Network` and `Convolutional Network (CNN)` for `pytorch`, also the `logistic regression` using `PCA`.

Our purpose is to create a model-supports archive to the public and explore the poosibilities for higher accuracy with a fast rate.

## Data
So far, we trained the models below and received the results below

| Model Name | Details | Accuracy | Code |
| -- | -- | -- | -- |
| `DeepNN` | 6 layers, 784 nodes, softmax, 5 RELUs, ReduceLROnPlateu scheduler, Adam optimizer, and augmentation | 98.2-98.7 % | [DeepNN](train_torch/mnist_train_torch_NN-deep.py) |
| `SimpleCNN` | 2 Convlayers, 2 Fully Connected Layers, 2 batch normalization, 2RELUs, Adam optimizer, Forward method pass, ReduceLROnPlateu scheduler | 99.0-99.2 % | [SimpleCNN](train_torch/mnist_train_torch_CNN-fast.py) |
| `EnhancedCNN` |  3 Convlayers, 2 Fully Connected Layers, 3 batch normalization, 3RELUs, 2 $\times$ 2 Pooling Layer, Adam optimizer, StepLR scheduler (step size = 5, gamma = 0.5) | 99.4-99.55% | [EnhancedCNN](train_torch/nmist_train_torch_CNN-deep.py) |
| `Logistic Regression` | PCA (500 Components), test size = 0.2, max iteration = 1000 | ~92% | [Logistic Reg](mnist_data_train.py) |
| `KNN` | PCA (100 Components), Euclidean (L2) | 94.6-95.8 % | `REDACTED` |
| `KNN` | PCA (100 Components), Euclidean (L2), deskewing, noise removal, blurring | 96.3-96.6 % | [KNN-EucL2](KNN/mnist_data_train_EucL2.py) |
| `KNN` | PCA (100 Components), Minkowski (L3) | 95.9 % | [KNN-MkskL3](KNN/mnist_data_train_MkskiL3.py) |

There are more that yet to be found, stay tuned for the newest testing result.
For `EnhancedCNN` we created a test script that tests the dataset with different step sizes and gamma values. Feel free to play with them! [test_file](mnist_data_test.py)

## Reference
The MNIST Library: http://yann.lecun.com/exdb/mnist/
