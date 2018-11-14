#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import os.path as osp
import sys
sys.path.append("..")
reload(sys)  
sys.setdefaultencoding('utf8')   # about python2.7 encoding problem whtch not in python3
from tqdm import tqdm
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
from config import config
from lib.utils.pyt_utils import  ensure_dir ,link_file
from preprocess import Seed_TrainPre
from network import GAP_HighRes
from lib.datasets import Seed_VOC
from get_label import get_label_batch

# lr * ( 1-idx/ all)^0.9 
# idx bigger, lr smaller
def poly_lr(current_idx):
	lr = config.lr * ((1 - float(current_idx) / (config.epoches * config.epoch_num + 0.0)) ** config.lr_power)   
	return lr

# keep_rate from 1.0 to config.keep_rate
def poly_keep(current_idx):
    keep_rate = 1.0 - (1.0 - config.keep_rate) * float(current_idx) / (config.epoches * config.epoch_num + 0.0)
    drop_rate = 1.0 - keep_rate 
    
    return drop_rate

#torch.cuda.set_device(0)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
data_setting = {'img_root': config.img_root_folder,
                'gt_root': config.gt_root_folder,
                'train_source': config.train_source,
                'eval_source': config.eval_source} 

train_preprocess = Seed_TrainPre(config.image_mean, config.image_std, config.target_size)
trainloader = data.DataLoader(Seed_VOC(data_setting, "train", train_preprocess),  batch_size = config.batch_size, num_workers = config.num_workers, shuffle = True, pin_memory = True)

model = GAP_HighRes(config, split = 'train', pretrained = True)
model = nn.DataParallel(model, device_ids = config.gpu_ids, output_device = config.gpu_ids[0])
model.cuda()  
#for idx , para in enumerate(model.parameters()):
#    print(idx)
#    print(para.size())

criterion = nn.BCEWithLogitsLoss(size_average = True) 
criterion.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

cudnn.benchmark = True  # to use cudnn pretrained parameters
model.train()           # this has any effect only on modules such as Dropout or BatchNorm.
drop_rate = 0.0
for epoch in range(config.epoches):
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.epoch_num), file=sys.stdout, bar_format=bar_format) # equal to single batch number
    data_iters = iter(trainloader)

    # idx : step in every single batch
    for idx in pbar:
        minibatch = data_iters.next()
        imgs = minibatch['data']   # (16, 3, 321, 321)
        gts = minibatch['mask']
        batch_files = minibatch['batch_files']
        
        imgs = Variable(imgs.cuda())
        gts = Variable(gts.cuda())
        
        labels = get_label_batch(batch_files)
        labels = torch.from_numpy(np.ascontiguousarray(labels)).float()          
        labels = Variable(labels.cuda())
# forward
        output = model(imgs, drop_rate)   
        
        loss = criterion(output, labels  )   # float for output ,  long for gts
# backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_idx = epoch * config.epoch_num + idx +1
        lr = poly_lr(current_idx)
        drop_rate = poly_keep(current_idx)
        
        optimizer = torch.optim.SGD(model.parameters() , lr = lr, momentum = config.bn_momentum, weight_decay = config.weight_decay)
        optimizer.zero_grad()
    
 #print results
        print_str = 'Epoch{}/{}'.format(epoch, config.epoches) \
                    + '   ' \
                    + 'Iter{}/{}'.format(idx + 1, config.epoch_num) \
                    + '   ' \
                    + 'loss = %.2f' % loss.data \
                    + '   ' \
                    + 'lr = %.1e' % lr \
                    + '\n'       
        pbar.set_description(print_str)
          
    
    ensure_dir(config.snapshot_dir)
    if epoch % config.snapshot_iter == 0:
        ensure_dir(config.snapshot_dir)                 
        if not osp.exists(config.log_dir_link):
            link_file(config.log_dir, config.log_dir_link)
        current_epoch_checkpoint = osp.join(config.snapshot_dir, 'epoch-{}.pth'.format(epoch))  
        torch.save(model.state_dict(), current_epoch_checkpoint)                            
        last_epoch_checkpoint = osp.join(config.snapshot_dir, 'epoch-last.pth')       
        link_file(current_epoch_checkpoint, last_epoch_checkpoint)  

    for layer_idx, para in enumerate(model.parameters()):
        if layer_idx == 36:   
            np.savetxt(osp.join(config.snapshot_dir, 'fc_weight.txt'), 
                           para.cpu().detach().numpy())
        if layer_idx == 37:
            np.savetxt(osp.join(config.snapshot_dir, 'fc_bias.txt'), 
                           para.cpu().detach().numpy())

final_epoch_checkpoint = osp.join(config.snapshot_dir, "epoch-{}.pth".format(epoch))
torch.save(model.state_dict(), final_epoch_checkpoint)
last_epoch_checkpoint = osp.join(config.snapshot_dir, 'epoch-last.pth')
link_file(final_epoch_checkpoint, last_epoch_checkpoint)

for layer_idx, para in enumerate(model.parameters()):
    if layer_idx == 36:   
        np.savetxt(osp.join(config.snapshot_dir, 'fc_weight.txt'), 
                           para.cpu().detach().numpy())
    if layer_idx == 37:   
        np.savetxt(osp.join(config.snapshot_dir, 'fc_bias.txt'), 
                           para.cpu().detach().numpy())
       
#Visual_loss ( np.array(loss_list), osp.join(config.snapshot_dir, 'loss.png'))




















