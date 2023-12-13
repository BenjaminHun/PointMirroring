import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
import os
from torchvision.io import read_image
from torch.utils.data import DataLoader
import numpy as npl
import swin_transformer as sw

input = torch.zeros(10,3,128,128, dtype=torch.float)
a=nn.Conv2d(3, 6, 3, 1, 'same')
b=nn.MaxPool2d(3, stride=2,padding='same')
input=a(input)
input=b(input)
print(input.shape)
'''
     self.swinT=sw.SwinTransformer(img_size=(100,100),patch_size=5,window_size=5, in_chans=4,depths=[1])
        self.convStack = nn.Sequential(
            nn.ConvTranspose2d(96, 48,3,padding=0,output_padding=2,stride=5),
            nn.ReLU(),
            nn.Conv2d(48, 12, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 4, 3, padding=1),
            )

'''
