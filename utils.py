import os
import yaml
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def load_config(config_name, config_path):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config


class DiceLoss(nn.Module):
    def __init__(self, sigmoid=False, epsilon=1, positive_weight=1):
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


def plot_pred(preds, inputs, labels=None):
    pred_out = preds[0][0].detach().numpy()
    background = inputs[0][2].detach().numpy()
    plt.imshow(background, cmap='Greys', alpha=1)
    plt.imshow(pred_out, 
            cmap='YlOrRd',
            alpha=pred_out*.5)
    if labels != None:
        labels = labels > 0
        labels = labels.type(torch.int8)
        label_out = labels[0][0].numpy()
        plt.imshow(label_out, 
           cmap='RdYlBu', 
           alpha=label_out*.5)
        plt.xlabel('Prediction = Red, True Label = Blue',)
    else:
        plt.xlabel('Prediction = Red',)
    plt.title('Carotid Artery Segmentation')
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    
    plt.show()


if __name__ == "__main__":
    pass