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
from utils import DiceLoss, load_config
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from utils import DiceLoss


class carotidSegmentation():
    def __init__(self, model='unet_3'):
        config_path='./config/'
        config_file=f'{model}.yaml'
        config = load_config(config_file, config_path)
        batch_size = 1
        
        self._image_transforms = torch.nn.Sequential(
            transforms.CenterCrop(512),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            )
        
        self.net = UNet(
            in_channels=config['in_channels'], 
            n_classes=config['n_classes'], 
            depth=config['depth'], 
            batch_norm=config['batch_norm'], 
            padding=config['padding'], 
            up_mode=config['up_mode'])
        self.net.load_state_dict(torch.load(f'models/{model}.pth'))
        self.net.eval()
    
    def get_image(self,image_loc):
        image = read_image(image_loc)
        image = image.float()
        image = image.unsqueeze(0)
        image = self._image_transforms(image)
        return image

    def get_label(self, image_loc):
        mask_loc = image_loc.replace("US images", "Expert mask images")
        mask = read_image(mask_loc)
        mask = mask.float()
        mask = mask.unsqueeze(0)
        mask = self._image_transforms(mask)
        mask = mask > 0
        mask = mask.type(torch.int8)
        
        return mask

    def predict(self,image_loc):
        image = self.get_image(image_loc)
        preds = self.net(image)
        return preds

    def eval(self, image_loc):
        pred = self.predict(image_loc)
        label = self.get_label(image_loc)
        loss = DiceLoss()
        loss_out = loss(pred, label)
        return loss_out.item()
    
    def plot_pred(self, image_loc, labels=False):
        image = self.get_image(image_loc)
        preds = self.predict(image_loc)
        pred_out = preds[0][0].detach().numpy()
        background = image[0][2].detach().numpy()
        plt.imshow(background, cmap='Greys', alpha=1)
        plt.imshow(pred_out, 
                cmap='YlOrRd',
                alpha=pred_out*.5)
        if labels != False:
            label_out = self.get_label(image_loc)
            label_out = label_out[0][0]#.numpy()
            plt.imshow(label_out, 
            cmap='RdYlBu', 
            alpha=label_out*.5)
            dice_loss = round(self.eval(image_loc), 4)
            plt.xlabel(f'Prediction = Red, True Label = Blue \n Dice Loss: {dice_loss}')
        else:
            plt.xlabel('Prediction = Red',)
        plt.title('Carotid Artery Segmentation')
        plt.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 
        
        plt.show()


if __name__ == "__main__":
    image_loc = 'data/Common Carotid Artery Ultrasound Images/US images/202201121748100022VAS_slice_2319.png'
    seg_model = carotidSegmentation()
    test_out = seg_model.predict(image_loc)
    print(test_out.shape)
    seg_model.plot_pred(image_loc, labels=True)