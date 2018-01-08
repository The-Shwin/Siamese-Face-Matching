import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch import cat

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),  # 1
            nn.ReLU(inplace=True),  # 2 ReLU (in-place)
            nn.BatchNorm2d(64),  # 3 batch normalization (64 features)
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4 Max pooling layer
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),  # 5
            nn.ReLU(inplace=True),  # 6
            nn.BatchNorm2d(128),  # 7
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 9
            nn.ReLU(inplace=True),  # 10
            nn.BatchNorm2d(256),  # 11
            nn.MaxPool2d(kernel_size=2, stride=2),  # 12
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # 13
            nn.ReLU(inplace=True),  # 14
            nn.BatchNorm2d(512),  # 15
        )
        # The 16th layer is flatten (not a torch.nn layer)
        self.fully_connect = nn.Sequential(
            nn.Linear(16*16*512, 1024),  # 17
            nn.ReLU(inplace=True),  # 18
            nn.BatchNorm2d(1024) # 19
        )


    def single_run(self, img):
        image = self.cnn(img)
        image = image.view(image.size()[0], -1)  # 16th layer
        image = self.fully_connect(image)
        return image

    def forward(self, image1, image2):
        out_image1 = self.single_run(image1)
        out_image2 = self.single_run(image2)
        euclidean = nn.functional.pairwise_distance(out_image1, out_image2)
        return euclidean
