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
import numpy as np
import swin_transformer as sw
import math as m
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class CustomImageDataset(Dataset):

    def __init__(self, img_dir, annotations_file):
        super().__init__()
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        noisedImgPath = os.path.join(
            self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(noisedImgPath)[:3, :, :]
        groundTruthImgPath = os.path.join(
            self.img_dir, self.img_labels.iloc[idx, 1])
        # Remove alpha layer
        label = read_image(groundTruthImgPath)[:3, :, :]
        global device
        if device == "cuda":
            return torch.tensor(image, dtype=torch.float32).cuda(), torch.tensor(label, dtype=torch.float32).cuda()
        else:
            return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

outputPath="NeuralNetworkWithSwinT2"
trainDataset = CustomImageDataset("train/", "labels.txt")
testDataset = CustomImageDataset("test/", "labels.txt")

trainDataloader = DataLoader(trainDataset, batch_size=10, shuffle=True)
testDataloader = DataLoader(testDataset, batch_size=10, shuffle=True)

trainFeatures, trainLabels = next(iter(trainDataloader))

print(f"Feature batch shape: {trainFeatures.size()}")
print(f"Labels batch shape: {trainLabels.size()}")

print(f"Using {device} device")


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

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
            128, 128), patch_size=4, window_size=4, in_chans=3, depths=[16], embed_dim=96, drop_rate=0.3)
        self.convStack = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 3, padding=1,
                               output_padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 3, padding=1,
                               output_padding=1, stride=2),
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


class NeuralNetworkWithSwinT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.swinT = sw.SwinTransformer(img_size=(
            128, 128), patch_size=4, window_size=4, in_chans=3, depths=[8], embed_dim=96, drop_rate=0.3)
        self.swinT2 = sw.SwinTransformer(img_size=(
            32, 32), patch_size=4, window_size=4, in_chans=96, depths=[8], embed_dim=96, drop_rate=0.3)

        self.convStack = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 3, padding=1,
                               output_padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 3, padding=1,
                               output_padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 3, padding=1,
                               output_padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 3, padding=1,
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


#model=ConvNetwork()
#model=NeuralNetworkWithSwinT()
model = NeuralNetworkWithSwinT2()


if device == "cuda":
    model = model.cuda()

learningRate = 1e-3
batchSize = 10
epochs = 300
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def trainLoop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    i = 0
    for batch, (x, y) in enumerate(dataloader):
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 5 == 0:
            loss, current = loss.item(), (batch+1)*len(x)
            with open(outputPath +"/"+"results.txt","a",encoding="utf-8") as f:
                f.write(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]\n")
            x_base = normalize(x[0].permute(1, 2, 0).contiguous().cpu().detach().numpy())
            y_pred = normalize(pred[0].permute(1, 2, 0).contiguous().cpu().detach().numpy())
            y_gt = normalize(y[0].permute(1, 2, 0).contiguous().cpu().detach().numpy())
            border=np.ones([1,128,3])
            plt.imsave(outputPath +"/"+
                       str(i)+"_result.png", np.concatenate((y_pred,border,y_gt,border,x_base),0))

            i += 1


def testLoop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    i = 0
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            if i % 10 == 0:
                x_base = normalize(x[0].permute(1, 2, 0).contiguous().cpu().detach().numpy())
                y_pred = normalize(pred[0].permute(1, 2, 0).contiguous().cpu().detach().numpy())
                y_gt = normalize(y[0].permute(1, 2, 0).contiguous().cpu().detach().numpy())
                border=np.ones([1,128,3])
                plt.imsave(outputPath +"/"+
                       str(i)+"_test_result.png", np.concatenate((y_pred,border,y_gt,border,x_base),0))
                i += 1

    test_loss /= num_batches
    with open(outputPath +"/"+"results.txt","a",encoding="utf-8") as f:
        f.write(f"Test error: \n Avg loss:{test_loss:>8f}\n")


for t in range(epochs):
    with open(outputPath +"/"+"results.txt","a",encoding="utf-8") as f:
        f.write(f"Epoch {t+1}\n------------------\n")
    trainLoop(trainDataloader, model, loss, optimizer)
    testLoop(testDataloader, model, loss)
