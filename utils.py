import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, sigmoid=True):
        super(DiceLoss, self).__init__()
        self.sigmoid = sigmoid

    def forward(self, preds, labels, epsilon=1):
        if self.sigmoid == True:
            preds = F.sigmoid(preds)   
            
        diceLabel = labels.sum(dim=[1,2,3])
        dicePrediction = preds.sum(dim=[1,2,3])
        diceCorrect = (preds * labels).sum(dim=[1,2,3])

        diceRatio = (2. * diceCorrect + epsilon)/(dicePrediction + diceLabel + epsilon)       
        return (1 - diceRatio).mean()

if __name__ == "__main__":
    pass