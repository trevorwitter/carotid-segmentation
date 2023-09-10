import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, labels, epsilon=1):
        preds = F.sigmoid(preds)       
        #flatten label and prediction tensors
        preds = preds.view(-1)
        labels = labels.view(-1)
        intersection = (preds * labels).sum()                            
        dice = (2.*intersection + epsilon)/(preds.sum() + labels.sum() + epsilon)  
        return 1 - dice
    

if __name__ == "__main__":
    pass