import cv2 
import pygetwindow as gw
from PIL import ImageGrab

import numpy as np
import torch
import torch.nn as nn
from train_torch.nmist_train_torch_CNN_deep import EnhancedCNN

import threading
from flask import Flask, render_template

# Load model ----------------------------------------------------------------------
model = EnhancedCNN()
model.load_state_dict(torch.load('models/cnn-deep-model.pth', weights_only=True))
model.eval()

