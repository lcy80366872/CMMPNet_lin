import torch
import torch.nn as nn
import cv2
import numpy as np


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode =='con_ce':
            return self.ConLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, #ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def ConLoss(self, logit, target):
        # loss = torch.mean(torch.sum(-target * torch.log(F.softmax(logit, dim=1)), dim=1))
        # loss = torch.mean(torch.sum(-target * nn.LogSoftmax()(logit), dim=1))
        loss = nn.BCEWithLogitsLoss()(logit, target)
        # loss = nn.BCELoss()(logit, target)
        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.25):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, #ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1.0 
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def resize(self, y_true, h, w):
        b = y_true.shape[0]
        y = np.zeros((b,  h,w,y_true.shape[1]))
        
        y_true = np.array(y_true.cpu())
        for id in range(b):
#             y1 = y_true[id,:,:,:].transpose(1,2,0)
#             y[id,:,:,:] = cv2.resize(y1, (h, w))
#             print(y1.shape)
            a = cv2.resize(y1, (h, w))
#             print(a.ndim)
            if a.ndim == 2:
                a=np.expand_dims(a,axis=-1)
            print(y[id,:,:,:].shape)
            print(y.shape)
            y[id, :, :, :]=a
            print(a.shape)

#             print(y.shape)
            y=y.transpose(0,2,3,1)
        return torch.Tensor(y)
        
    def __call__(self, y_true, y_pred):
        # the ground_truth map is resized to the resolution of the predicted map during training
        if y_true.shape[2] != y_pred.shape[2] or y_true.shape[3] != y_pred.shape[3]:
            print(y_true.shape)
            y_true = self.resize(y_true, y_pred.shape[2], y_pred.shape[3]).cuda()
           
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b
