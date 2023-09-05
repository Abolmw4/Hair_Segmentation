import torch
import torch.nn as nn
import math


class BCELoss2d(nn.Module):
    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, predict: torch.Tensor, target: torch.tensor) -> torch.tensor:
        """
        this function calculate BCE loss between output from model in mask
        :param predict: output tensor from model (batch, channel or class, heigh, width)
        :param target: mask of each input image (batch, channel or class, heigh, width)
        :return: return 1D tensor
        """
        predict = predict  # make 1D tensor
        target = target  # make 1D tensor
        return self.bce_loss(predict, target)


def dice_coeff(predict, target):
    smooth = 1e-5
    batch_size: int = predict.size(0)
    predict = (predict > 0.5).float()
    m1 = predict.view(batch_size, -1)  # this is tensor with size (batch size, channel*heigh*width)
    m2 = target.view(batch_size, -1)  # this is tensor with size (batch size, channel*heigh*width)
    intersection = (m1 * m2).sum(-1)
    return ((2.0 * intersection + smooth) / (m1.sum(-1) + m2.sum(-1) + smooth)).mean()


def get_IoU(outputs, labels):
    EPS = 1e-6
    outputs = outputs.int()
    labels = labels.int()

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0

    iou = (intersection + EPS) / (union + EPS)  # We smooth our devision to avoid 0/0

    return iou.mean()


def calculate_overlap_metrics(gt, pred):
    output = pred.view(-1, ).to("cuda")
    target = gt.view(-1, ).float().to('cuda')

    tp = torch.sum(output * target)  # TP
    fp = torch.sum(output * (1 - target))  # FP
    fn = torch.sum((1 - output) * target)  # FN
    tn = torch.sum((1 - output) * (1 - target))  # TN

    pixel_acc = (tp + tn + 1e-5) / (tp + tn + fp + fn + 1e-5)

    return pixel_acc
