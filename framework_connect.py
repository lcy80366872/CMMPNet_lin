import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import  numpy as np
from tqdm import tqdm
from utils.metrics import IoU
from loss import dice_bce_loss,SegmentationLosses
# from loss import SegmentationLosses
import copy
import numpy


def pre_general(output, out_connect, out_connect_d1):

    out_connect = out_connect.data.cpu().numpy()
    pred_connect = np.mean(out_connect, axis=1)[:,np.newaxis, :, :]
    pred_connect[pred_connect < 0.9] = 0
    pred_connect[pred_connect >= 0.9] = 1


    out_connect_d1 = out_connect_d1.data.cpu().numpy()
    pred_connect_d1 = np.mean(out_connect_d1, axis=1)[:,np.newaxis, :, :]

    pred_connect_d1[pred_connect_d1 < 2.0] = 0
    pred_connect_d1[pred_connect_d1 >= 2.0] = 1


    pred = output.data.cpu().numpy()

    pred[pred > 0.1] = 1
    pred[pred < 0.1] = 0

    su = pred + pred_connect + pred_connect_d1
    su[su > 0] = 1

    return torch.Tensor(su)
class Solver:
    def __init__(self, net, optimizer, dataset):
        self.net = torch.nn.DataParallel(net.cuda(), device_ids=list(range(torch.cuda.device_count())))
        
        self.optimizer = optimizer
        self.dataset = dataset

        self.loss = dice_bce_loss()
        self.criterion_con = SegmentationLosses(weight=None, cuda=True).build_loss(mode='con_ce')

        self.metrics = IoU(threshold=0.5)
        self.old_lr = optimizer.param_groups[0]["lr"]
        
    def set_input(self, img_batch, mask_batch=None):
        self.img = img_batch
        self.mask = mask_batch[:,0,:,:].unsqueeze(1)


        self.connect_label=mask_batch[:,1:10,:,:]
        self.connect_d1_label = mask_batch[:, 10:, :, :]
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
                    self.connect_label=Variable(self.connect_label.cuda())
                    self.connect_d1_label = Variable(self.connect_d1_label.cuda())
            else:
                self.mask = Variable(self.mask.cuda())
                self.connect_label = Variable(self.connect_label.cuda())
                self.connect_d1_label = Variable(self.connect_d1_label.cuda())


    def optimize(self):
        self.net.train()
        self.data2cuda()
        
        self.optimizer.zero_grad()
        pred,connect,connect_d1 = self.net.forward(self.img)
        loss1 = self.loss(self.mask, pred)
        loss2 = self.loss(self.connect_label,connect)
        loss3 = self.loss(self.connect_d1_label,connect_d1 )
        lad = 0.2
        loss = loss1 + lad * (0.6 * loss2 + 0.4 * loss3)
        loss.backward()
        self.optimizer.step()
        pred=pre_general(pred,connect,connect_d1)
        batch_iou, intersection, union = self.metrics(self.mask, pred)
        return pred, loss.item(), batch_iou, intersection, union

    def test_batch(self):
        self.net.eval()
        self.data2cuda(volatile=True)
        pred,connect,connect_d1  = self.net.forward(self.img)
        loss1 = self.loss(self.mask, pred)
        loss2 = self.loss(self.connect_label,connect )
        loss3 = self.loss( self.connect_d1_label,connect_d1)
        lad = 0.2
        loss = loss1 + lad * (0.6 * loss2 + 0.4 * loss3)
        pred = pre_general(pred, connect, connect_d1)
        batch_iou, intersection, union = self.metrics(self.mask, pred)
        pred = pred.cpu().data.numpy().squeeze(1)
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

    def fit(self, epochs, no_optim_epochs=5):
        val_best_metrics = test_best_metrics = [0, 0]
        no_optim = 0

        for epoch in range(1, epochs + 1):
            print(f"epoch {epoch}/{epochs}")
            
            train_loss, train_metrics = self.fit_one_epoch(self.train_dl,      mode='training')
            val_loss, val_metrics     = self.fit_one_epoch(self.validation_dl, mode='val')
            test_loss, test_metrics   = self.fit_one_epoch(self.test_dl,       mode='testing')

            if val_best_metrics[1] < val_metrics[1]:
               val_best_metrics  = val_metrics
               test_best_metrics = test_metrics
               val_best_net = copy.deepcopy(self.solver.net.state_dict())
               epoch_val = epoch
               no_optim = 0
            else:
               no_optim += 1

            if no_optim > no_optim_epochs:
                if self.solver.old_lr < 1e-8:
                    print('early stop at {epoch} epoch')
                    break
                else:
                    no_optim = 0
                    self.solver.update_lr(ratio=10.0)
                    
            print(f'train_loss: {train_loss:.4f} train_metrics: {train_metrics}')
            print(f'  val_loss: {val_loss:.4f}   val_metrics:   {val_metrics}')
            print(f' test_loss: {test_loss:.4f}  test_metrics:  {test_metrics}')
            print('current best epoch:', epoch_val, ',val g_iou:', val_best_metrics[1], ',test g_iou:', test_best_metrics[1])
            print('epoch finished')
            print()

        print("############ Final IoU Results ############")
        print('selected epoch: ', epoch_val)
        print(' val set: A_IOU ', val_best_metrics[0],  ', G_IOU ', val_best_metrics[1])
        print('test set: A_IOU ', test_best_metrics[0], ', G_IOU ', test_best_metrics[1])
        torch.save(val_best_net, os.path.join(self.save_path, f"epoch{epoch_val}_val{val_best_metrics[1]:.4f}_test{test_best_metrics[1]:.4f}.pth"))

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
            if mode=='training':
                pred_map, iter_loss, batch_iou, samples_intersection, samples_union = self.solver.optimize()
            else:
                pred_map, iter_loss, batch_iou, samples_intersection, samples_union = self.solver.test_batch()
                
            epoch_loss += iter_loss
            progress_bar.set_description(f'{mode} iter: {i} loss: {iter_loss:.4f}')
            
            local_batch_iou += batch_iou

            samples_intersection = samples_intersection.cpu().data.numpy()
            samples_union        = samples_union.cpu().data.numpy()
            for sample_id in range(len(samples_intersection)):
                if samples_union[sample_id] == 0: # the IOU is ignored when its union is 0
                   continue
                intersection.append(samples_intersection[sample_id])
                union.append(samples_union[sample_id])
                   
        intersection = numpy.array(intersection)
        union        = numpy.array(union)
        
        '''
        In the code[1] of paper[1], average_iou is the mean of the IoU of all batches. 
        For a fair comparison, we follow code[1] to compute the average_iou in our paper.
        However, more strictly, average_iou should be the mean of the IoU of all samples, i.e., average_iou = (intersection/union).mean()
         
        I recommend using global_iou
        
        paper[1]: Leveraging Crowdsourced GPS Data for Road Extraction from Aerial Imagery, CVPR 2019
        code[1]: https://github.com/suniique/Leveraging-Crowdsourced-GPS-Data-for-Road-Extraction-from-Aerial-Imagery/blob/master/framework.py#L106
        '''
        average_iou = local_batch_iou / iter_num
        #average_iou = (intersection/union).mean()
        
        global_iou  = intersection.sum()/union.sum()
        metrics = [average_iou, global_iou]

        return epoch_loss, metrics
        
        
