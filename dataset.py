import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_image

class CarotidDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="data/Common Carotid Artery Ultrasound Images", crop=True):
        self.root_dir = root_dir
        image_loc = os.path.join(root_dir,"US images")
        self.image_paths = [os.path.join(image_loc, x) for x in os.listdir(image_loc)]
        mask_loc = os.path.join(root_dir,"Expert mask images")
        self.mask_paths = [os.path.join(mask_loc, x) for x in os.listdir(mask_loc)]
        self._transforms = torch.nn.Sequential(
            transforms.CenterCrop(512),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            )
        self.crop = crop

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx])
        image = image.float()
        label = read_image(self.mask_paths[idx])
        if self.crop == True:
            image, label = self.random_crop(image, label)
        else:
            image = self._transforms(image)
            label = self._transforms(label)
        return image, label
    
    def get_center(self, label):
        """Returns center coordinates of carotid artery mask"""
        indices = torch.nonzero(label)
        row_min = torch.min(indices,dim=0).values[1].item()
        row_max = torch.max(indices,dim=0).values[1].item()

        col_min = torch.min(indices,dim=0).values[2].item()
        col_max = torch.max(indices,dim=0).values[2].item()

        row_center = (row_min + row_max)//2
        col_center = (col_min + col_max)//2
        return (row_center, col_center)

    def random_crop(self, img, label, height=256, width=256):
        """Returns random crop subsection of image; Consider using when
           mask makes up low % of overall image"""
        row_center, col_center = self.get_center(label)
        vert_split = np.random.randint(150)
        hor_split = np.random.randint(150)

        top = top = row_center - vert_split
        left = col_center - hor_split
        img = transforms.functional.crop(img, top=top, left=left, height=height, width=height)
        label = transforms.functional.crop(label, top=top, left=left, height=height, width=height)
        return img, label
       

if __name__ == "__main__":
    data = CarotidDataset()
    print(f"Images: {data.__len__()}")
    print("-"*10)
    x, y = next(iter(data))
    print(f"Input image shape: {x.shape}")
    print(f"Input Mask shape:  {y.shape}")
    print("-"*10)

