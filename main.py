import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from model.GCRAN_TS import GCRAN_TS as Network
from model.train import Trainer
from lib.dataloader import get_dataloader
import random

def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def print_model_parameters(model, only_num = True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')

#*************************************************************************#
Mode = 'train'
DEBUG = 'False'
DATASET = 'NORTH_PACIFIC_1'
DEVICE = 'cuda:0'
MODEL = 'RGCN_PSA'

#get configuration
config_file = './{}_{}.conf'.format(MODEL, DATASET)
config = configparser.ConfigParser()
config.read(config_file)

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default=Mode, type=str)
args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--cuda', default=True, type=bool)

#data
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--daily_window', default=config['data']['daily_window'], type=int)
args.add_argument('--yearly_window', default=config['data']['yearly_window'], type=int)
args.add_argument('--shift_window', default=config['data']['shift_window'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)

#model
args.add_argument('--dim_in', default=config['model']['dim_in'], type=int)
args.add_argument('--dim_encoder_hidden', default=config['model']['dim_encoder_hidden'], type=int)
args.add_argument('--dim_attention', default=config['model']['dim_attention'], type=int)
args.add_argument('--dim_k', default=config['model']['dim_k'], type=int)
args.add_argument('--dim_v', default=config['model']['dim_v'], type=int)
args.add_argument('--nums_head', default=config['model']['nums_head'], type=int)
args.add_argument('--dim_short_hidden', default=config['model']['dim_short_hidden'], type=int)
args.add_argument('--dim_gcn_in', default=config['model']['dim_gcn_in'], type=int)
args.add_argument('--dim_graph', default=config['model']['dim_graph'], type=int)
args.add_argument('--num_clusters', default=config['model']['num_clusters'], type=int)
args.add_argument('--cheb_k', default=config['model']['cheb_k'], type=int)
args.add_argument('--dim_gcn_hidden', default=config['model']['dim_gcn_hidden'], type=int)
args.add_argument('--dim_clusters_hidden', default=config['model']['dim_clusters_hidden'], type=int)
args.add_argument('--dim_tsa_hidden', default=config['model']['dim_tsa_hidden'], type=int)
args.add_argument('--dim_long_hidden', default=config['model']['dim_long_hidden'], type=int)
args.add_argument('--dim_out', default=config['model']['dim_out'], type=int)

#train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')

#log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)
args = args.parse_args()
init_seed(args.seed)
if torch.cuda.is_available() and args.device != 'cpu':
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'
print('device:', args.device)

#load dataset 训练集
train_loader, val_loader, test_loader, adj_dataloader, scaler = get_dataloader(args, normalizer = args.normalizer)

#init model
model = Network(args, adj_dataloader)
print(args.device)
model = model.to(args.device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

#init loss function, optimizer
if args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

#config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'experiments', args.dataset, current_time)
args.log_dir = log_dir

#start training
trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler, args, lr_scheduler=lr_scheduler)
if args.mode == 'train':
    #print("train------------------")
    trainer.train()
elif args.mode == 'test':
    #print("test------------------")
    model.load_state_dict(torch.load('../pre-trained/{}.pth'.format(args.dataset)))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
else:
    #print("error------------------")
    raise ValueError
