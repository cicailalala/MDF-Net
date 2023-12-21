import torch.nn as nn
import torch


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, pred, target):
        num = target.size(0)
        probs = torch.sigmoid(pred)
        m1 = probs.view(num, -1)
        m2 = target.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        #score = 1 - score.sum() / num
        score = - torch.log(score.sum() / num)
        return score


# def soft_dice(pred, target, size_average=True, batch_average=True):
#     loss_f = SoftDiceLoss()
#     return loss_f(pred, target)