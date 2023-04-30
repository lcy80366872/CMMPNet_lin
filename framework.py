import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import os
from tqdm import tqdm
from utils.metrics import IoU
from loss import dice_bce_loss
import copy
import numpy
from networks.DirectionNet import DirectionNet
from networks.SGCN import Sobel


import torch_dct as DCT

def show_sobal(img,channels):
    sobal=Sobel(channels, channels)
    img_sobal=sobal(img)
    print('img', img.shape)
    print('sobla',img_sobal)
    plt.subplot(1, 2, 1)
    plt.imshow(img[0,0, :, :].cpu())
    plt.subplot(1, 2, 2)
    plt.imshow(img_sobal[0,:,:].cpu())
    # plt.subplot(1, 3, 3)
    # plt.imshow(b[0, 0, :, :].cpu())
    plt.show()

def L1_penalty(var):
    return torch.abs(var).sum()
class Solver:
    def __init__(self, net, optimizer, dataset):
        # self.net = torch.nn.DataParallel(net.cuda(), device_ids=list(range(torch.cuda.device_count())))
        self.net=net.cuda()
#         self.net_direction=DirectionNet().cuda()
        self.optimizer = optimizer
        self.dataset = dataset
        self.loss1 =dice_bce_loss(ssim=True)
        self.loss = dice_bce_loss(ssim=True)
        self.metrics = IoU(threshold=0.5)
        self.old_lr = optimizer.param_groups[0]["lr"]
    def resize(self, y_true, h, w):
        b = y_true.shape[0]
        y = numpy.zeros((b, h, w, y_true.shape[1]))
        # print('y_t:', y_true.shape)
        y_true = numpy.array(y_true.cpu())
        for id in range(b):
            y1 = y_true[id, :, :, :].transpose(1, 2, 0)
            # print('y1:', y1.shape)
            a = cv2.resize(y1, (h, w))

            if a.ndim == 2:
                a = numpy.expand_dims(a, axis=-1)
            # print('a:', a.shape)
            y[id, :, :, :] = a
        # print(y.shape)
        y = y.transpose(0, 3, 1,2)

        return torch.Tensor(y)


    def set_input(self, img_batch, mask_batch=None):
        self.img = img_batch
        self.mask = mask_batch

    def data2cuda(self, volatile=False):
        if volatile:
            with torch.no_grad():
                self.img = Variable(self.img.cuda())
        else:
            self.img = Variable(self.img.cuda())

        if self.mask is not None:
            if volatile:
                with torch.no_grad():
                    self.mask = Variable(self.mask.cuda())
            else:
                self.mask = Variable(self.mask.cuda())

    def optimize(self):
        self.net.train()
        self.data2cuda()

        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)

        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()

        batch_iou, intersection, union = self.metrics(self.mask, pred)
        return pred, loss.item(), batch_iou, intersection, union

    #
    # def optimize_exchange(self):
    #     self.net.train()
    #     self.data2cuda()
    #
    #     self.optimizer.zero_grad()
    #     outs = self.net.forward(self.img)
    #     slim_params = []
    #     for name, param in self.net.named_parameters():
    #         if param.requires_grad and name.endswith('weight') and 'bn2' in name:
    #             if len(slim_params) % 2 == 0:
    #                 slim_params.append(param[:len(param) // 2])
    #             else:
    #                 slim_params.append(param[len(param) // 2:])
    #     loss=0
    #     for output in outs:
    #         soft_output = nn.LogSoftmax()(output)
    #         loss += self.loss(self.mask,soft_output)
    #         L1_norm = sum([L1_penalty(m).cuda() for m in slim_params])
    #         lamda =2e-4
    #         loss += lamda * L1_norm  # this is actually counted for len(outputs) times
    #
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     batch_iou, intersection, union = self.metrics(self.mask, outs[0])
    #     return outs[0], loss.item(), batch_iou, intersection, union

    def optimize_exchange(self):
        self.net.train()
        self.data2cuda()

        self.optimizer.zero_grad()
        # mask = self.resize(self.mask, 512, 512).cuda()
        # direct_mask=self.net_direction.forward(mask)

        pred = self.net.forward(self.img)
        slim_params = []
        for name, param in self.net.named_parameters():
            if param.requires_grad and name.endswith('weight') and 'bn2' in name:
                if len(slim_params) % 2 == 0:
                    slim_params.append(param[:len(param) // 2])
                else:
                    slim_params.append(param[len(param) // 2:])

        loss = self.loss(self.mask,pred)

        # loss += self.loss1(self.mask, pred)
        # loss += self.loss(self.mask, pred1)
        # loss +=0.2*self.loss_direction(direct_pred,direct_mask)
        L1_norm = sum([L1_penalty(m).cuda() for m in slim_params])
        lamda =2e-4
        loss += lamda * L1_norm  # this is actually counted for len(outputs) times


        loss.backward()
        self.optimizer.step()

        batch_iou, intersection, union = self.metrics(self.mask, pred)
        return pred, loss.item(), batch_iou, intersection, union

    def test_batch(self):
        self.net.eval()
        self.data2cuda(volatile=True)
        # mask = self.resize(self.mask, 512, 512).cuda()
        # direct_mask = self.net_direction.forward(mask)
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
           # loss += self.loss(self.mask, pred1)
        # loss +=0.2*self.loss_direction(direct_pred,direct_mask)

        batch_iou, intersection, union = self.metrics(self.mask, pred)
        pred = pred.cpu().data.numpy().squeeze(1)
        return pred, loss.item(), batch_iou, intersection, union
    def test_batch_exchange(self):
        self.net.eval()
        self.data2cuda(volatile=True)

        outs = self.net.forward(self.img)
        slim_params = []
        for name, param in self.net.named_parameters():
            if param.requires_grad and name.endswith('weight') and 'bn2' in name:
                if len(slim_params) % 2 == 0:
                    slim_params.append(param[:len(param) // 2])
                else:
                    slim_params.append(param[len(param) // 2:])
        loss = 0
        for output in outs:
            soft_output = nn.LogSoftmax()(output)
            # Compute loss and backpropagate
            loss += self.loss(self.mask, soft_output)
            L1_norm = sum([L1_penalty(m).cuda() for m in slim_params])
            lamda = 2e-4
            loss += lamda * L1_norm  # this is actually counted for len(outputs) times
        # loss = self.loss(self.mask, pred)
        # L1_norm = sum([L1_penalty(m).cuda() for m in slim_params])
        # lamda = 2e-4
        # loss += lamda * L1_norm  # this is actually counted for len(outputs) times

        batch_iou, intersection, union = self.metrics(self.mask, outs[0])
        pred = outs[0].cpu().data.numpy().squeeze(1)
        return pred, loss.item(), batch_iou, intersection, union

    def update_lr(self, ratio=1.0):
        new_lr = self.old_lr / ratio
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        print("==> update learning rate: %f -> %f" % (self.old_lr, new_lr))
        self.old_lr = new_lr


class Framework:
    def __init__(self, *args, **kwargs):
        self.solver = Solver(*args, **kwargs)

    def set_train_dl(self, dataloader):
        self.train_dl = dataloader

    def set_validation_dl(self, dataloader):
        self.validation_dl = dataloader

    def set_test_dl(self, dataloader):
        self.test_dl = dataloader

    def set_save_path(self, save_path):
        self.save_path = save_path

    def fit(self, epochs, no_optim_epochs=10):
        val_best_metrics = test_best_metrics = [0, 0]
        no_optim = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.solver.optimizer, T_max=epochs,
                                                               verbose=True)
        for epoch in range(1, epochs + 1):
            print(f"epoch {epoch}/{epochs}")

            train_loss, train_metrics = self.fit_one_epoch(self.train_dl, mode='training')
            val_loss, val_metrics = self.fit_one_epoch(self.validation_dl, mode='val')
            test_loss, test_metrics = self.fit_one_epoch(self.test_dl, mode='testing')
            if val_best_metrics[1] < val_metrics[1]:
                val_best_metrics = val_metrics
                test_best_metrics = test_metrics
                val_best_net = copy.deepcopy(self.solver.net.state_dict())
                epoch_val = epoch
                no_optim = 0
            else:
                no_optim += 1
            scheduler.step()
            if no_optim > no_optim_epochs:
                if self.solver.old_lr < 1e-8:
                    print('early stop at {epoch} epoch')
                    break
                else:
                    no_optim = 0
                    self.solver.update_lr(ratio=5.0)

            print(f'train_loss: {train_loss:.4f} train_metrics: {train_metrics}')
            print(f'  val_loss: {val_loss:.4f}   val_metrics:   {val_metrics}')
            print(f' test_loss: {test_loss:.4f}  test_metrics:  {test_metrics}')
            print('current best epoch:', epoch_val, ',val g_iou:', val_best_metrics[1], ',test g_iou:',
                  test_best_metrics[1])
            print('epoch finished')
            print()

        print("############ Final IoU Results ############")
        print('selected epoch: ', epoch_val)
        print(' val set: A_IOU ', val_best_metrics[0], ', G_IOU ', val_best_metrics[1])
        print('test set: A_IOU ', test_best_metrics[0], ', G_IOU ', test_best_metrics[1])
        torch.save(val_best_net, os.path.join(self.save_path,
                                              f"epoch{epoch_val}_val{val_best_metrics[1]:.4f}_test{test_best_metrics[1]:.4f}.pth"))

    def fit_one_epoch(self, dataloader, mode='training'):
        epoch_loss = 0.0
        local_batch_iou = 0.0
        intersection = []
        union = []

        dataloader_iter = iter(dataloader)
        iter_num = len(dataloader_iter)
        progress_bar = tqdm(enumerate(dataloader_iter), total=iter_num)

        for i, (img, mask) in progress_bar:
            self.solver.set_input(img, mask)
            # print('img_data:',img.shape)
            if mode == 'training':
                pred_map, iter_loss, batch_iou, samples_intersection, samples_union = self.solver.optimize_exchange()
            else:
                pred_map, iter_loss, batch_iou, samples_intersection, samples_union = self.solver.test_batch()

            epoch_loss += iter_loss
            progress_bar.set_description(f'{mode} iter: {i} loss: {iter_loss:.4f}')

            local_batch_iou += batch_iou

            samples_intersection = samples_intersection.cpu().data.numpy()
            samples_union = samples_union.cpu().data.numpy()
            for sample_id in range(len(samples_intersection)):
                if samples_union[sample_id] == 0:  # the IOU is ignored when its union is 0
                    continue
                intersection.append(samples_intersection[sample_id])
                union.append(samples_union[sample_id])

        intersection = numpy.array(intersection)
        union = numpy.array(union)

        '''
        In the code[1] of paper[1], average_iou is the mean of the IoU of all batches. 
        For a fair comparison, we follow code[1] to compute the average_iou in our paper.
        However, more strictly, average_iou should be the mean of the IoU of all samples, i.e., average_iou = (intersection/union).mean()

        I recommend using global_iou

        paper[1]: Leveraging Crowdsourced GPS Data for Road Extraction from Aerial Imagery, CVPR 2019
        code[1]: https://github.com/suniique/Leveraging-Crowdsourced-GPS-Data-for-Road-Extraction-from-Aerial-Imagery/blob/master/framework.py#L106
        '''
        average_iou = local_batch_iou / iter_num
        # average_iou = (intersection/union).mean()

        global_iou = intersection.sum() / union.sum()
        metrics = [average_iou, global_iou]

        return epoch_loss, metrics

