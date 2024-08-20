import cv2 
import numpy as np
import torch
import torch.nn as nn
from train_torch.mnist_train_torch_CNN_deep import EnhancedCNN
import pygetwindow

model = EnhancedCNN()
model.load_state_dict(torch.load('models/cnn-deep-model.pth', weights_only=True))

