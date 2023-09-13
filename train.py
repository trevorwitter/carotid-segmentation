import os
import json
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
from dataset import CarotidDataset
from unet import UNet
from utils import DiceLoss
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default=1, type=int, help="Number of workers")
    parser.add_argument("--batch_size", default=32, type=int, help="Number of images in each batch")
    parser.add_argument("--gpu", default=True, type=bool, help="Train on GPU True/False")
    parser.add_argument("--epochs", default=1, type=int, help="Number of training epochs")
    parser.add_argument("--warm_start", default=False, type=bool, help="Loads trained model")
    return parser.parse_args()

def training_loop(net, trainloader, gpu=False, batch_size=8, epochs=1):
    if gpu == False:
        device = torch.device("cpu")
    elif gpu == True:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Training on {device.type}")
    model_name = "unet"
    tb = SummaryWriter(f'runs/{model_name}')
    
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    step_count = 0
    loss = DiceLoss()
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs - 1}')
        print('-' * 10)
        net.train()
        net = net.to(device)
        train_running_loss = 0.0
        train_running_corrects = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = net(inputs)
            preds = F.pad(preds, (3,2,7,6))
            train_loss = loss(preds, labels)
            train_loss.backward()
            optimizer.step()
            step_count += 1
            train_running_loss += train_loss.item()
            print(f"batch {i} loss: {train_loss.item()}")
    train_loss = train_running_loss / len(trainloader.dataset)
    tb.add_scalar('Loss/Training',
                train_loss,
                epoch)
    PATH = f'./models/{model_name}.pth'
    torch.save(net.state_dict(), PATH)

    print(f'Training complete - model saved to {PATH}')
    tb.close()


def main(args):
    data = CarotidDataset()
    dataloader = torch.utils.data.DataLoader(
        data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers)
    net = UNet(in_channels=3, n_classes=1, padding=True, up_mode='upsample')
    dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    training_loop(net, dataloader, gpu=args.gpu, batch_size=args.batch_size, epochs=args.gpu)


if __name__ == "__main__":
    args = arg_parse()
    main(args)