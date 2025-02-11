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
def pre_general_test(output, out_connect, out_connect_d1):
    out_connect_full = []
    out_connect = out_connect.data.cpu().numpy()
    out_connect_full.append(out_connect[0, ...])
    out_connect_full.append(out_connect[1, :, ::-1, :])
    out_connect_full.append(out_connect[2, :, :, ::-1])
    out_connect_full.append(out_connect[3, :, ::-1, ::-1])
    out_connect_full = np.asarray(out_connect_full).mean(axis=0)[np.newaxis, :, :, :]
    pred_connect = np.sum(out_connect_full, axis=1)
    pred_connect[pred_connect < 0.9] = 0
    pred_connect[pred_connect >= 0.9] = 1

    out_connect_d1_full = []
    out_connect_d1 = out_connect_d1.data.cpu().numpy()
    out_connect_d1_full.append(out_connect_d1[0, ...])
    out_connect_d1_full.append(out_connect_d1[1, :, ::-1, :])
    out_connect_d1_full.append(out_connect_d1[2, :, :, ::-1])
    out_connect_d1_full.append(out_connect_d1[3, :, ::-1, ::-1])
    out_connect_d1_full = np.asarray(out_connect_d1_full).mean(axis=0)[np.newaxis, :, :, :]
    pred_connect_d1 = np.sum(out_connect_d1_full, axis=1)
    pred_connect_d1[pred_connect_d1 < 2.0] = 0
    pred_connect_d1[pred_connect_d1 >= 2.0] = 1

    pred_full = []
    pred = output
    # target_n = target.cpu().numpy()
    pred_full.append(pred[0, ...])
    pred_full.append(pred[1, :, ::-1, :])
    pred_full.append(pred[2, :, :, ::-1])
    pred_full.append(pred[3, :, ::-1, ::-1])
    pred_full = np.asarray(pred_full).mean(axis=0)

    pred_full[pred_full > 0.1] = 1
    pred_full[pred_full < 0.1] = 0

    su = pred_full + pred_connect + pred_connect_d1
    su[su > 0] = 1
    print('su_shape:',su.shape)
    return torch.Tensor(su)
def replace_direction_value(a,b):
    c=torch.min(a,b)
    a1=a>0.5
    b1=b>0.5
    c1=(a1==b1)
    a2=a<0.5
    b2=b<0.5
    c2=(a2==b2)
    c=torch.where(c1,a*b,c)
    c = torch.where(c2, a * b, c)
    return c
   
def direction_process(general_mask):

    img = general_mask
    # img[img >= 0.5] = 1
    # img[img < 0.5] = 0
    shp = img.shape   #n*c*h*w
    # print('shap:',shp)
    # print('img:',img)
    # if img.ndim == 3:
    #     img = np.expand_dims(img, axis=1)

    img_pad = torch.zeros([shp[0],shp[1],shp[2] + 2, shp[3] + 2])
    # print('img:', img_pad.shape)
    img_pad[:,:,1:-1, 1:-1] = img
    # print('img:',img_pad.shape)
    # connect = torch.zeros([shp[0],9,shp[2], shp[3]])
    #roll参数分别为输入、滚动距离和滚动维度
    c1=torch.roll(img_pad,[1,1] ,[2,3])[:,:,1:-1,1:-1]
    c2=torch.roll(img_pad,[1,0] ,[2,3])[:,:,1:-1,1:-1]
    c3=torch.roll(img_pad,[1,-1] ,[2,3])[:,:,1:-1,1:-1]
    c4=torch.roll(img_pad,[0,1] ,[2,3])[:,:,1:-1,1:-1]
    c5=torch.roll(img_pad,[0,-1] ,[2,3])[:,:,1:-1,1:-1]
    c6=torch.roll(img_pad,[-1,1] ,[2,3])[:,:,1:-1,1:-1]
    c7=torch.roll(img_pad,[-1,0] ,[2,3])[:,:,1:-1,1:-1]
    c8=torch.roll(img_pad,[-1,-1] ,[2,3])[:,:,1:-1,1:-1]
    c1 = c1.cuda()
    c2 = c2.cuda()
    c3 = c3.cuda()
    c4 = c4.cuda()
    c5 = c5.cuda()
    c6 = c6.cuda()
    c7 = c7.cuda()
    c8 = c8.cuda()
#     connect = torch.cat((img , replace_direction_value(c1,img)), 1)
#     connect = torch.cat((connect , replace_direction_value(c2,img)), 1)
#     connect = torch.cat((connect, replace_direction_value(c3, img)), 1)
#     connect = torch.cat((connect, replace_direction_value(c4, img)), 1)
#     connect = torch.cat((connect, replace_direction_value(c5, img)), 1)
#     connect = torch.cat((connect, replace_direction_value(c6, img)), 1)
#     connect = torch.cat((connect, replace_direction_value(c7, img)), 1)
#     connect = torch.cat((connect, replace_direction_value(c8, img)), 1)
    connect = torch.cat((img, c1 * img), 1)
    connect = torch.cat((connect, c2 * img), 1)
    connect = torch.cat((connect, c3 * img), 1)
    connect = torch.cat((connect, c4 * img), 1)
    connect = torch.cat((connect, c5 * img), 1)
    connect = torch.cat((connect, c6 * img), 1)
    connect = torch.cat((connect, c7 * img), 1)
    connect = torch.cat((connect, c8 * img), 1)


    # connect = torch.cat((img,torch.where(img > 0.5, c1*img, img)),1)
    # connect =  torch.cat((connect,torch.where(img > 0.5, c2*img, img)),1)
    # connect =  torch.cat((connect,torch.where(img > 0.5, c3*img, img)),1)
    # connect = torch.cat((connect,torch.where(img > 0.5, c4*img, img)),1)
    # connect= torch.cat((connect,torch.where(img > 0.5, c5*img, img)),1)
    # connect = torch.cat((connect,torch.where(img > 0.5, c6*img, img)),1)
    # connect = torch.cat((connect,torch.where(img > 0.5, c7*img, img)),1)
    # connect = torch.cat((connect,torch.where(img > 0.5, c8*img, img)),1)
    # print("connect:", connect.shape)
    return connect
class Solver:
    def __init__(self, net, optimizer, dataset):
        self.net = torch.nn.DataParallel(net.cuda(), device_ids=list(range(torch.cuda.device_count())))
        
        self.optimizer = optimizer
        self.dataset = dataset

        self.loss = dice_bce_loss()
        self.loss_con=nn.BCEWithLogitsLoss()
        self.criterion_con = SegmentationLosses(weight=None, cuda=True).build_loss(mode='con_ce')

        self.metrics = IoU(threshold=0.5)
        self.old_lr = optimizer.param_groups[0]["lr"]
        
    def set_input(self, img_batch, mask_batch=None):
        self.img = img_batch
        self.mask = mask_batch[:,0,:,:].unsqueeze(1)
        self.connect_label=mask_batch[:,1:10,:,:]
        # self.connect_d1_label = mask_batch[:, 10:, :, :]
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
                    # self.connect_d1_label = Variable(self.connect_d1_label.cuda())
            else:
                self.mask = Variable(self.mask.cuda())
                self.connect_label = Variable(self.connect_label.cuda())
                # self.connect_d1_label = Variable(self.connect_d1_label.cuda())


    def optimize(self):
        self.net.train()
        self.data2cuda()
        
        self.optimizer.zero_grad()
        # pred,connect,connect_d1 = self.net.forward(self.img)
        pred = self.net.forward(self.img)
        #这里clone一下就是因为直接用pred进行操作那么在传播时会有问题
        pred1=pred.clone()
        loss1 = self.loss(self.mask, pred)
        connect=direction_process(pred1).cuda()
        # print('connect:', connect.shape)
        loss2 = self.loss(self.connect_label,connect)
        # loss3 = self.loss(self.connect_d1_label,connect_d1 )
        lad = 0.2
        loss=loss1+lad*loss2
        # loss = loss1 + lad * (0.6 * loss2 + 0.4 * loss3)
        loss.backward()
        self.optimizer.step()
        # pred=pre_general(pred,connect,connect_d1)
        batch_iou, intersection, union = self.metrics(self.mask, pred)
        return pred, loss.item(), batch_iou, intersection, union

    def test_batch(self):
        self.net.eval()
        self.data2cuda(volatile=True)

        pred = self.net.forward(self.img)
        loss1 = self.loss(self.mask, pred)
        pred1 = pred.clone()
        connect = direction_process(pred1).cuda()
        loss2 = self.loss(self.connect_label,connect)
        lad = 0.2
        loss = loss1 + lad * loss2

        batch_iou, intersection, union = self.metrics(self.mask, pred)
        pred = pred.cpu().data.numpy().squeeze(1)

        # pred,connect,connect_d1  = self.net.forward(self.img)
        # loss1 = self.loss(self.mask, pred)
        # loss2 = self.loss(self.connect_label,connect )
        # # loss3 = self.loss( self.connect_d1_label,connect_d1)
        # lad = 0.2
        # loss=loss1+lad*loss2
        # # loss = loss1 + lad * (0.6 * loss2 + 0.4 * loss3)
        # img = self.img.cpu().numpy()
        # image1 = img[:, :, ::-1, :]
        # image2 = img[:, :, :, ::-1]
        # image3 = img[:, :, ::-1, ::-1]
        # img = np.concatenate((img, image1, image2, image3), axis=0)
        # img = torch.from_numpy(img).float()
        # pred, connect, connect_d1 = self.net.forward(img)
        # pred = pre_general_test(pred, connect, connect_d1)
        # batch_iou, intersection, union = self.metrics(self.mask, pred)
        # pred = pred.cpu().data.numpy().squeeze(1)
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

    def fit(self, epochs, no_optim_epochs=3):
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
                    self.solver.update_lr(ratio=5.0)
                    
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
        # print('iter_num:',iter_num)
        progress_bar = tqdm(enumerate(dataloader_iter), total=iter_num)
        
        for i, (img, mask) in progress_bar:
            # print('mask:',mask.shape)
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
        
        
