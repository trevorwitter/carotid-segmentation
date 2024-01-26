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
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='unet_1', type=str, help='Model Experiment')
    return parser.parse_args()

def eval(test_loader, net, gpu=True):
    if gpu == False:
        device = torch.device('cpu')
    elif gpu == True:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    loss = DiceLoss()
    net = net.to(device)
    net.eval()
    test_step_count = 0
    for i, data in enumerate(test_loader, 0):
            test_running_loss = 0.0
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds = net(inputs)
            test_loss = loss(preds, labels)
            test_step_count += 1
            test_running_loss += test_loss.item()
    test_loss = test_running_loss / len(test_loader.dataset)
    return test_loss

def main(model):
    config_path='./config/'
    config_file=f'{model}.yaml'
    config = load_config(config_file, config_path)
    data = CarotidDataset(crop=config['crop'])
    generator = torch.Generator().manual_seed(42)
    train_data, val_data, test_data = torch.utils.data.random_split(data, [0.8, 0.1, 0.1], generator=generator)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=config['batch_size'], 
        shuffle=config['shuffle'], 
        num_workers=config['num_workers'],
        generator=generator
        )

    net = UNet(in_channels=config['in_channels'], 
                n_classes=config['n_classes'], 
                depth=config['depth'], 
                batch_norm=config['batch_norm'], 
                padding=config['padding'], 
                up_mode=config['up_mode'])
    net.load_state_dict(torch.load(f'models/{model}.pth'))
    
    test_loss = eval(test_loader, net)
    print(f'Test Set Dice Loss: {test_loss}')


if __name__ == "__main__":
    args = arg_parse()
    main(args)