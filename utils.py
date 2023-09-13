import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, sigmoid=True, epsilon=1, positive_weight=8):
        super(DiceLoss, self).__init__()
        self.sigmoid = sigmoid
        self.epsilon = epsilon
        self.positive_weight = positive_weight

    def dice(self, preds, labels, epsilon):     
        diceLabel = labels.sum(dim=[1,2,3])
        dicePrediction = preds.sum(dim=[1,2,3])
        diceCorrect = (preds * labels).sum(dim=[1,2,3])

        diceRatio = (2. * diceCorrect + epsilon)/(dicePrediction + diceLabel + epsilon)       
        return (1 - diceRatio).mean()

    def forward(self, preds, labels):
        if self.sigmoid == True:
            preds = F.sigmoid(preds)
        labels = labels > 0    

        dice_loss = self.dice(preds, labels, self.epsilon)
        fn_loss = self.dice(preds * labels, labels, self.epsilon)
        return dice_loss + fn_loss * self.positive_weight

if __name__ == "__main__":
    pass