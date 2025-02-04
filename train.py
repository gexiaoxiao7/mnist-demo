import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt

# mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# linux / windows
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
