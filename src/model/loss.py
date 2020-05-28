import torch
import torch.nn as nn


class MAELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MAELoss, self).__init__()
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        t = torch.abs(predict - target)
        if self.reduction == 'mean':
            return t.mean(), t
        elif self.reduction == 'sum':
            return t.sum(), t
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
