import cv2 
import numpy as np
import torch
import torch.nn as nn
from train_torch.nmist_train_torch_CNN_deep import EnhancedCNN

model = EnhancedCNN()
model.load_state_dict(torch.load('cnn-deep-model.pth'))


