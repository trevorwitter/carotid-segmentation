import os
import torch
from torchvision import transforms
from torchvision.io import read_image

class CarotidDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="data/Common Carotid Artery Ultrasound Images"):
        self.root_dir = root_dir
        image_loc = os.path.join(root_dir,"US images")
        self.image_paths = [os.path.join(image_loc, x) for x in os.listdir(image_loc)]
        mask_loc = os.path.join(root_dir,"Expert mask images")
        self.mask_paths = [os.path.join(mask_loc, x) for x in os.listdir(mask_loc)]
        self.transforms = torch.nn.Sequential(
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            )

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx])
        image = image.float()
        label = read_image(self.mask_paths[idx])
        #return self.transforms(image), label
        return image, label

if __name__ == "__main__":
    data = CarotidDataset()
    print(f"Images: {data.__len__()}")
    print("-"*10)
    x, y = next(iter(data))
    print(f"Input image shape: {x.shape}")
    print(f"Input Mask shape:  {y.shape}")
    print("-"*10)

