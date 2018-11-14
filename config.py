#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict  # visit dict like characteristics

c = edict()
config = c

c.gpu_ids = [0,1]
c.num_gpu = len(c.gpu_ids)
c.seed = 3
#root -dir
# repo_name is the user name , repo_name = 'hhhhhhhhhh'
c.repo_name = 'hhhhhhhhhh'
# return the user path ,         abs_dir = '/home/hhhhhhhhhh'
c.abs_dir = osp.realpath('.')
# osp.sep = path split symbol,     this_dir= 'hhhhhhhhhh'
c.this_dir = c.abs_dir.split(osp.sep)[-1]
# index is 6 , plus 10 in hhhhhhhhhh, then construct root_dir, a little unnecessary
                                   # root_dir = '/home/hhhhhhhhhh'
c.root_dir = c.abs_dir[:c.abs_dir.index(c.repo_name) + len(c.repo_name)]
                                   # log_dir = '/home/hhhhhhhhhh/log/hhhhhhhhhh'
c.log_dir = osp.abspath(osp.join(c.root_dir, 'log', c.this_dir))
                                  # log_dir_link '/home/hhhhhhhhhh/log'
c.log_dir_link = osp.join(c.abs_dir, 'log')
                                   # snapshot_dir = '/home/hhhhhhhhhh/log/hhhhhhhhhh/snapshot'
#c.snapshot_dir = osp.abspath(osp.join(c.log_dir, "snapshot"))
c.snapshot_dir = osp.abspath(osp.join(c.log_dir, "snapshot-11.14-linear-nodrop"))


exp_time = time.strftime('&Y_%m_&d_&H_%M_%S', time.localtime())
c.log_file = c.log_dir+ '/log_' + exp_time +'.log'
c.link_log_file = c.log_file +'/log_last.log'
c.val_log_file = c.log_dir + '/val' + exp_time + '.log'
c.linnk_val_log_file = c.log_dir + '/val_last.log'


#  data dir
c.img_root_folder = "/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/JPEGImages"
c.gt_root_folder = "/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/SegmentationClass"
c.sal_root_folder = "/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/val_initial_saliency"
c.seed_root_folder = "/home/hhhhhhhhhh/desktop/seed/seed每次结果/log/seedcilcle/seed0.5val"

c.train_source = "/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/list/train_aug.txt"
c.eval_source = "/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/list/val.txt" 
                         #
c.test_source = ""                         #
c.is_test = False                           #

# image parameter
c.num_classes = 20
c.background = 0
c.image_mean = np.array([0.485, 0.456, 0.406])
c.image_std = np.array([0.229 , 0.224, 0.224])
c.target_size = 320
c.image_height = 320
c.image_width =320
c.num_train_imgs = 10582
c.num_eval_imgs = 1449                 # 

# train config
c.keep_rate = 0.9
c.lr = 0.05    # 7e-2        # 0.001
c.lr_power = 0.9
c.momentum = 0.9
c.weight_decay = 5e-4
c.batch_size = 4*c.num_gpu
c.epoches = 60
c.epoch_num = int(np.ceil(c.num_train_imgs // c.batch_size))
# how many batch in one epoch
c.bn_momentum = 0.1
c.fix_bn = False                             #
c.snapshot_iter = 5
c.record_info_iter = 20
c.display_iter = 50
c.train_scale_array = [0.5, 0.75, 1, 1.5, 1.75, 2]
c.num_workers = 10

# eval config                         #
c.eval_iter = 30
c.eval_scale_array = [1, ]
c.eval_flip = False
c.eval_base_size = 512
c.eval_crop_size = 512

# test seed
c.threshold = 0.2
#c.weight_path = '/home/hhhhhhhhhh/log/seed/snapshot'
#c.load_path = '/home/hhhhhhhhhh/log/seed/snapshot/epoch-{}.pth'.format(c.epoches - 1 )
#
c.weight_path = '/home/hhhhhhhhhh/log/seed/snapshot-11.14-linear-drop-feature'
c.load_path = '/home/hhhhhhhhhh/log/seed/snapshot-11.14-linear-drop-feature/epoch-5.pth'
# visualisation
c.visual_path =  '/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/results/GAP_ALL'
c.alpha = 0.625

# proposal
c.prop_m_path = '/home/hhhhhhhhhh/desktop/dataset/MCG_proposal/sbd_voc_all_mfile'
c.prop_path = '/home/hhhhhhhhhh/desktop/dataset/MCG_proposal/val_proposal'


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0,path)

add_path(osp.join(c.root_dir, 'lib'))      # dynamically set searching path priorority
