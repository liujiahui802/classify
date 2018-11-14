#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import time
from PIL import Image
import scipy.misc as mic

import torch
import numpy as np
import torch.utils.data as data


class Seed_BaseDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None):
        super(Seed_BaseDataset, self).__init__()
        self._split_name = split_name
        self._img_path = setting['img_root']
        self._gt_path = setting['gt_root']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._file_names = self._get_file_names(split_name)
        self.preprocess = preprocess

    def __len__(self):
        return len(self._file_names)

    def __getitem__(self, index):
    	# _file_name is the train or val data file name list , including [img, gt] fo each item
        names = self._file_names[index]
        img_path = self._img_path + '/'+ names[0]
        gt_path = self._gt_path + '/' + names[1]
        item_name = names[1].split("/")[-1].split(".")[0]    
        # item_name is file name exclude postfix

        # b g r order in channel dimension ,    h w c in shape
#        img = self._open_image(img_path)
##        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE)
#        img = img[:, :, ::-1]  # from BGR to RGB, for torch process
        gt = np.array(Image.open(gt_path).convert(mode = 'P'), dtype = np.uint8)
        img = np.array(Image.open(img_path), dtype = np.uint8)
        
        if self.preprocess is not None:# val for seedcycle and test for seed rectangle
            if self._split_name in ['train', 'val']:
                img, gt = self.preprocess(img, gt)
            if self._split_name is 'test':
                img, gt, img_dict = self.preprocess(img, gt)

        img = img.transpose(2,0,1)
		# from (512, 512,3) to (3, 512, 512)
        
        img = torch.from_numpy(np.ascontiguousarray(img)).float()   # in memory continuously
        gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
        if self._split_name in ['train' , 'val']:
            return dict(data = img, mask = gt , batch_files = str(item_name), n = len(self._file_names))
        if self._split_name is 'test':
            return dict(data=img, mask=gt, batch_files=str(item_name), n=len(self._file_names) , img_dict = img_dict)

        # to get train or val data list file name in [img, gt ] format
    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val', 'test']    
        source = self._train_source
        if split_name in['val', 'test']:
            source = self._eval_source
        # source is a path to train data list or val data list
        file_names = []
        with open(source) as f:
            files = f.readlines()
        # to let shuffle different in different time
        time_seed = int(time.time() * 1000) % 4294967295
        np.random.seed(time_seed)
        np.random.shuffle(files)
        #??????????????????????  according to different train list format
        for item in files:
            item = item.strip()
            item = item.split(' ')
            img_name = item[0].split('/')[-1]
            gt_name = item[1].split('/')[-1]
            file_names.append([img_name, gt_name])  # jpg & png

        return file_names

    def get_length(self):
        return self.__len__()


class Sal_BaseDataset(data.Dataset):
    def __init__(self, config, split_name, preprocess=None):
        super(Sal_BaseDataset, self).__init__()
        self._split_name = split_name
        self._img_path = config.img_root_folder
        self._gt_path = config.gt_root_folder
        self._sal_path = config.sal_root_folder
        self._train_source = config.train_source
        self._eval_source = config.eval_source
        self._file_names = self._get_file_names(split_name)
        self.preprocess = preprocess

    def __len__(self):
        return len(self._file_names)

    def __getitem__(self, index):
    	# _file_name is the train or val data file name list , including [img, gt] fo each item
        # item_name is file name exclude postfix
        names = self._file_names[index]
        img_path = self._img_path + '/'+ names[0]
        gt_path = self._gt_path + '/' + names[1]
        sal_path = self._sal_path + '/' + names[1]
        item_name = names[1].split("/")[-1].split(".")[0]    

        gt = np.array(Image.open(gt_path).convert(mode = 'P'), dtype = np.uint8)
        img = np.array(Image.open(img_path), dtype = np.uint8)
        sal = np.array(Image.open(sal_path), dtype = np.uint8) 
        sal = sal[:, (sal.shape[1]/2) : sal.shape[1]]
         
        if self.preprocess is not None:
                img, gt, sal = self.preprocess(img, gt, sal)
                img1 = mic.imresize(img, 0.5)
                img2 = mic.imresize(img, 0.75)
                img3 = img
                img = img.transpose(2,0,1)
                img1 = img1.transpose(2,0,1)
                img2 = img2.transpose(2,0,1)
                img3 = img3.transpose(2,0,1)
		        # from (512, 512,3) to (3, 512, 512)
                img = torch.from_numpy(np.ascontiguousarray(img)).float()
                img1 = torch.from_numpy(np.ascontiguousarray(img1)).float()   # in memory continuously
                img2 = torch.from_numpy(np.ascontiguousarray(img2)).float()
                img3 = torch.from_numpy(np.ascontiguousarray(img3)).float()   
                
        sal = torch.from_numpy(np.ascontiguousarray(sal)).long()
        gt = torch.from_numpy(np.ascontiguousarray(gt)).long()

        return dict(data = img, data1 = img1, data2 = img2, data3 = img3, mask = gt , sal = sal , batch_files = str(item_name), 
                        n = len(self._file_names))


        # to get train or val data list file name in [img, gt ] format
    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val', 'test']    
        source = self._train_source
        if split_name in['val', 'test']:
            source = self._eval_source
        file_names = []
        with open(source) as f:
            files = f.readlines()
        time_seed = int(time.time() * 1000) % 4294967295
        np.random.seed(time_seed)
        np.random.shuffle(files)
        for item in files:
            item = item.strip()
            item = item.split(' ')
            img_name = item[0].split('/')[-1]
            gt_name = item[1].split('/')[-1]
            file_names.append([img_name, gt_name])  # jpg & png
        return file_names

    def get_length(self):
        return self.__len__()
