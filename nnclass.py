import torch.nn as nn
from torchvision.io import read_image
import swin_transformer as sw
import math as m

class NeuralNetworkWithSwinT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.swinT = sw.SwinTransformer(img_size=(
            128, 128), patch_size=4, window_size=4, in_chans=3, depths=[12], embed_dim=12, drop_rate=0.3)
        self.swinT2 = sw.SwinTransformer(img_size=(
            32, 32), patch_size=4, window_size=4, in_chans=12, depths=[12], embed_dim=24, drop_rate=0.3)

        self.convStack = nn.Sequential(
            
          
            nn.ConvTranspose2d(24, 16, 3, padding=1,
                               output_padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, padding=1,
                               output_padding=1, stride=2),
            nn.ConvTranspose2d(8, 4, 3, padding=1,
                               output_padding=1, stride=2),
            nn.ReLU(),
              nn.ConvTranspose2d(4, 3, 3, padding=1,
                               output_padding=1, stride=2),

        )

    def forward(self, x):
        x = self.swinT(x)
        H = 128
        W = 128
        B, HW, C = x.shape
        newH = int((H/(m.pow((H*W)/HW, 0.5))))
        newW = int((W/(m.pow((H*W)/HW, 0.5))))
        x = x.view(B, C, newH, newW)
        x = self.swinT2(x)
        H = 32
        W = 32
        B, HW, C = x.shape
        newH = int((H/(m.pow((H*W)/HW, 0.5))))
        newW = int((W/(m.pow((H*W)/HW, 0.5))))
        x = x.view(B, C, newH, newW)
        x = self.convStack(x)
        return x


    
class ConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.convStack = nn.Sequential(
            nn.Conv2d(3, 6, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(6, 12, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, 2, 1),
            nn.ReLU(),
           
            nn.ConvTranspose2d(48, 48, 3, padding=1,
                               output_padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 3, padding=1,
                               output_padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 3, padding=1,
                               output_padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 6, 3, padding=1,
                               output_padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 3, padding=1, output_padding=1, stride=2),
        )

    def forward(self, x):
        input = self.convStack(x)
        return input
    
class NeuralNetworkWithSwinT(nn.Module):
    def __init__(self):
        super().__init__()
        self.swinT = sw.SwinTransformer(img_size=(
            128, 128), patch_size=2, window_size=4, in_chans=3, depths=[12], embed_dim=96, drop_rate=0.3)
        self.convStack = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 3, padding=1,
                               output_padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 24, 3, padding=1,
                            ),
            nn.ReLU(),
            nn.Conv2d(24, 12, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 3, 3, padding=1),
        )

    def forward(self, x):
        x = self.swinT(x)
        H = 128
        W = 128
        B, HW, C = x.shape
        newH = int((H/(m.pow((H*W)/HW, 0.5))))
        newW = int((W/(m.pow((H*W)/HW, 0.5))))
        x = x.view(B, C, newH, newW)
        x = self.convStack(x)
        return x
    
class swinT3(nn.Module):
    def __init__(self):
        super().__init__()
        self.swinT = sw.SwinTransformer(img_size=(
            128, 128), patch_size=4, window_size=4, in_chans=3, depths=[12], embed_dim=12, drop_rate=0.3)
        self.swinT2 = sw.SwinTransformer(img_size=(
            32, 32), patch_size=4, window_size=4, in_chans=12, depths=[12], embed_dim=24, drop_rate=0.3)
        self.swinT3 = sw.SwinTransformer(img_size=(
            8, 8), patch_size=4, window_size=4, in_chans=24, depths=[12], embed_dim=48, drop_rate=0.3)

        self.convStack = nn.Sequential(
            nn.ConvTranspose2d(48, 36, 3, padding=1,
                               output_padding=1, stride=2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(36, 24, 3, padding=1,
                               output_padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 16, 3, padding=1,
                               output_padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, padding=1,
                               output_padding=1, stride=2),
            nn.ConvTranspose2d(8, 4, 3, padding=1,
                               output_padding=1, stride=2),
            nn.ReLU(),
              nn.ConvTranspose2d(4, 3, 3, padding=1,
                               output_padding=1, stride=2),

        )

    def forward(self, x):
        x = self.swinT(x)
        H = 128
        W = 128
        B, HW, C = x.shape
        newH = int((H/(m.pow((H*W)/HW, 0.5))))
        newW = int((W/(m.pow((H*W)/HW, 0.5))))
        x = x.view(B, C, newH, newW)
        x = self.swinT2(x)
        H = 32
        W = 32
        B, HW, C = x.shape
        newH = int((H/(m.pow((H*W)/HW, 0.5))))
        newW = int((W/(m.pow((H*W)/HW, 0.5))))
        x = x.view(B, C, newH, newW)
        x = self.swinT3(x)
        H = 8
        W = 8
        B, HW, C = x.shape
        newH = int((H/(m.pow((H*W)/HW, 0.5))))
        newW = int((W/(m.pow((H*W)/HW, 0.5))))
        x = x.view(B, C, newH, newW)
        x = self.convStack(x)
        return x