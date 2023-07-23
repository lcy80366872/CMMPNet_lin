import torch
import numpy as np
# from torchsummary import summary
import time
import torch.nn as nn
from torch.nn import init
import os
import sys
from torch_lr_finder import LRFinder
import cv2
from loss import dice_bce_loss
from utils.model_init import model_init
from framework import Framework
import wandb
# from framework_connect import Framework
# from utils.datasets_connect import prepare_Beijing_dataset, prepare_TLCGIS_dataset
from utils.datasets import prepare_Beijing_dataset, prepare_TLCGIS_dataset
from networks.exchange_dlink34net import DinkNet34_CMMPNet
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def get_model(model_name):
    if model_name == 'CMMPNet':
       model = DinkNet34_CMMPNet()
    else:
        print("[ERROR] can not find model ", model_name)
        assert(False)
    return model

def get_dataloader(args):
    if args.dataset =='BJRoad':
        train_ds, val_ds, test_ds = prepare_Beijing_dataset(args) 
    elif args.dataset == 'TLCGIS' or args.dataset.find('Porto') >= 0:
        train_ds, val_ds, test_ds = prepare_TLCGIS_dataset(args) 
    else:
        print("[ERROR] can not find dataset ", args.dataset)
        assert(False)  

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=True,  drop_last=False)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False, drop_last=False)
    test_dl  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False, drop_last=False)
    return train_dl, val_dl, test_dl
def predicting_road(img):
    net = get_model(args.model)
def pre_image(net):
    net.eval()
    img = cv2.imread('/kaggle/input/bjroad-connect/BJRoad/test/image/11_0_sat.png')
    img1 = cv2.imread('/kaggle/input/bjroad-connect/BJRoad/test/mask/11_0_mask.png', cv2.IMREAD_GRAYSCALE)
    img1 = np.expand_dims(img1, axis=2)
    print('img_shape', img.shape)
    print('img1_shape', img1.shape)
    img = np.concatenate([img, img1], axis=2)
    img = cv2.resize(img, (512, 512))
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    img = img[np.newaxis, :, :, :]
    img = torch.tensor(img)
    print('img_cat_shape', img.shape)
    img=img.cuda()
    with torch.no_grad():
        pred, connect, connect_d1 = net.forward(img)
    pred=pred.cpu().numpy()
    pred=np.squeeze(pred,axis=0).transpose(1,2,0)
    connect=connect.cpu().numpy()
    connect_d1 = connect_d1.cpu().numpy()
    print(pred.shape)
    # cv2.namedWindow("pred_img")
    cv2.imshow('img',pred)


def train_val_test(args):
    net = get_model(args.model)
#     with torch.no_grad():  # 蹇呴』鏈�
#         summary(net.to('cuda'), input_size=(4, 512, 512), batch_size=4)
    model_init(net, 'resnet34', 2, imagenet=True)
#     print(net)
    print('lr:',args.lr)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)
    # 澶歡pu寰楀埌鐨勬ā鍨媎ict鍓嶉潰浼氬姞module
#     new_state = {} 
#     state_dict = torch.load('/kaggle/input/connect-v2-model/connect_v2_epoch19_val0.6588_test0.6342.pth', map_location=torch.device('cpu'))
#     for key, value in state_dict.items():
#         new_state[key.replace('module.', '')] = value
#     net.load_state_dict(new_state)
    
    
#     pre_image(net)
    
    framework = Framework(net, optimizer, dataset=args.dataset)
    
    train_dl, val_dl, test_dl = get_dataloader(args)
    
#     criterion = dice_bce_loss()
#     lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
#     lr_finder.range_test(train_dl,val_loader=val_dl, end_lr=1, num_iter=100,step_mode="linear")
#     lr_finder.plot()  # to inspect the loss-learning rate graph
#     lr_finder.reset()  # to reset the model and optimizer to their initial state

    framework.set_train_dl(train_dl)
    framework.set_validation_dl(val_dl)
    framework.set_test_dl(test_dl)
    framework.set_save_path(WEIGHT_SAVE_DIR)
#     wandb.init(
#     # set the wandb project where this run will be logged
#     project="exchange_bnthrold",
#     config={
#     "lr": args.lr,
#     "model": args.model,
#     "batch_size": args.batch_size,
#     "gpu_ids":  args.gpu_ids ,
#     "epochs":  args.epochs,
#     "dataset": args.dataset
#         })
#     sweep_configuration = {
#         'method': 'random',
#         'name': 'sweep',
#         'metric': {
#             'goal': 'maximize',
#             'name': 'test_metrics'
#         },
#         'parameters': {
#             'batch_size': {'values': [2, 4, 8]},
#             'lr': {'max': 0.1, 'min': 0.0001}
#         }
#     }

#     sweep_id = wandb.sweep(sweep=sweep_configuration, project="exchange_search")
#     wandb.agent(sweep_id, function=framework.fit, count=8)

    framework.fit(epochs=args.epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CMMPNet')
    parser.add_argument('--lr',    type=float, default=2e-4)
    parser.add_argument('--name',  type=str, default='')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sat_dir',  type=str, default=r'F:\ML_data\remote_data\BJRoad\train_val\image')
    parser.add_argument('--mask_dir', type=str, default=r'F:\ML_data\remote_data\BJRoad\train_val\mask')
    parser.add_argument('--gps_dir',  type=str, default=r'F:\ML_data\remote_data\BJRoad\train_val\gps')
    parser.add_argument('--test_sat_dir',  type=str, default=r'F:\ML_data\remote_data\BJRoad\test\image')
    parser.add_argument('--test_mask_dir', type=str, default=r'F:\ML_data\remote_data\BJRoad\test\mask')
    parser.add_argument('--test_gps_dir',  type=str, default=r'F:\ML_data\remote_data\BJRoad\test\gps')
    parser.add_argument('--connect_dir1',  type=str, default=r'F:\ML_data\remote_data\BJRoad\train_val\connect_8_d1')
    parser.add_argument('--connect_dir2', type=str, default=r'F:\ML_data\remote_data\BJRoad\train_val\connect_8_d3')
    parser.add_argument('--test_connect_dir1', type=str,default=r'F:\ML_data\remote_data\BJRoad\test\connect_8_d1')
    parser.add_argument('--test_connect_dir2', type=str,default=r'F:\ML_data\remote_data\BJRoad\test\connect_8_d3')
    parser.add_argument('--lidar_dir',  type=str, default='')
    parser.add_argument('--split_train_val_test', type=str, default='')
    parser.add_argument('--weight_save_dir', type=str, default='./save_model')
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--use_gpu',  type=bool, default=True)
    parser.add_argument('--gpu_ids',  type=str, default='0')
    parser.add_argument('--workers',  type=int, default=0)
    parser.add_argument('--epochs',  type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--dataset', type=str, default='BJRoad')
    parser.add_argument('--down_scale', type=bool, default=False)
    args = parser.parse_args()

    if args.use_gpu:
        try:
            gpu_list = [int(s) for s in args.gpu_ids.split(',')]
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
        BATCH_SIZE = args.batch_size * len(gpu_list)
    else:
        BATCH_SIZE = args.batch_size
        
    WEIGHT_SAVE_DIR = os.path.join(args.weight_save_dir, f"{args.model}_{args.dataset}_"+time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())+"/")
    if not os.path.exists(WEIGHT_SAVE_DIR):
        os.makedirs(WEIGHT_SAVE_DIR)
    print("Log dir: ", WEIGHT_SAVE_DIR)
    
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(WEIGHT_SAVE_DIR+'train.log')
    

    train_val_test(args)
    print("[DONE] finished")
