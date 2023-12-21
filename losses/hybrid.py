from .focal import FocalLoss
from torch import nn
import torch

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)
        i_flat = prediction.view(-1)
        t_flat = target.view(-1)
        intersection = (i_flat * t_flat).sum()
        union = i_flat.sum() + t_flat.sum()
        dice = 1 - ((2. * intersection + self.smooth) / (union + self.smooth))
        return dice


class Hybrid_loss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(Hybrid_loss, self).__init__()
        self.bce = nn.BCELoss()
        self.Dice = DiceLoss()
        self.bce_weight = bce_weight
    def forward(self, prediction, target):
        bce = self.bce(torch.sigmoid(prediction).squeeze(1), target.type(prediction.dtype))
        dice = self.Dice(prediction, target)

        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
        return loss

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x

class IOULoss(nn.Module):
    def __init__(self, smooth=1.0, threshold=0.5):
        super(IOULoss, self).__init__()
        self.smooth = smooth
        self.threshold = threshold

    def forward(self, prediction, target):

        prediction = _threshold(prediction, threshold=self.threshold)

        intersection = torch.sum(target * prediction)
        union = torch.sum(target) + torch.sum(prediction) - intersection + self.smooth
        return (intersection + self.smooth) / union


class ACC(nn.Module):
    def __init__(self, smooth=1.0, threshold=0.5):
        super(ACC, self).__init__()
        self.smooth = smooth
        self.threshold = threshold

    def forward(self, prediction, target):
        prediction = _threshold(prediction, threshold=self.threshold)

        tp = torch.sum(target == prediction, dtype=prediction.dtype)
        score = tp / target.view(-1).shape[0]
        return score

class Fscore(nn.Module):
    def __init__(self, smooth=1.0, threshold=0.5, beta=1):
        super(Fscore, self).__init__()
        self.smooth = smooth
        self.threshold = threshold
        self.beta = beta

    def forward(self, prediction, target):
        prediction = _threshold(prediction, threshold=self.threshold)

        tp = torch.sum(target * prediction)
        fp = torch.sum(prediction) - tp
        fn = torch.sum(target) - tp

        score = ((1 + self.beta ** 2) * tp + self.smooth) \
                / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.smooth)
        return score

# if __name__ == '__main__':
#     print()