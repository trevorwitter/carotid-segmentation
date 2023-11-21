import os
import json
import yaml
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
from utils import DiceLoss, load_config
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='unet_1', type=str, help="Model Experiment")
    return parser.parse_args()

def training_loop(net, trainloader, gpu=False, batch_size=8, epochs=1, lr=0.001, positive_weight=1, model_name='unet'):
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
    tb = SummaryWriter(f'runs/{model_name}')
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    step_count = 0
    loss = DiceLoss(positive_weight=positive_weight)
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
            train_loss = loss(preds, labels)
            train_loss.backward()
            optimizer.step()
            step_count += 1
            train_running_loss += train_loss.item()
            print(f"batch {i} loss: {train_loss.item()}")
            tb.add_scalar('training running loss',
                            train_loss.item(),
                            step_count)
    train_loss = train_running_loss / len(trainloader.dataset)
    tb.add_scalar('Loss/Training',
                train_loss,
                epoch)
    PATH = f'./models/{model_name}.pth'
    torch.save(net.state_dict(), PATH)

    print(f'Training complete - model saved to {PATH}')
    tb.close()


def main(args):
    config_path='./config/'
    config_file=f'{args.model}.yaml'
    config = load_config(config_file, config_path)
    data = CarotidDataset(crop=config['crop'])
    dataloader = torch.utils.data.DataLoader(
        data, 
        batch_size=config['batch_size'], 
        shuffle=config['shuffle'], 
        num_workers=config['num_workers'],
        )
    net = UNet(
        in_channels=config['in_channels'], 
        n_classes=config['n_classes'], 
        depth=config['depth'], 
        batch_norm=config['batch_norm'], 
        padding=config['padding'], 
        up_mode=config['up_mode'],
        )
    training_loop(
        net, 
        dataloader, 
        gpu=config['gpu'], 
        batch_size=config['batch_size'], 
        epochs=config['epochs'],
        lr=config['learning_rate'],
        positive_weight=config['dice_positive_weight'],
        model_name=config['model_name'],
        )


if __name__ == "__main__":
    args = arg_parse()
    main(args)