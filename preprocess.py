#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import cv2 
import scipy.misc as mic
import numpy as np

from config import config
from lib.utils.img_utils import random_scale, random_mirror, normalize, generate_random_crop_pos, random_crop_pad_to_shape

class Seed_TrainPre(object):
    def __init__ (self, img_mean, img_std, target_size):
        self.img_mean = img_mean
        self.img_std  = img_std
        self.target_size = target_size
    def __call__(self, img, gt):
        img, gt = random_mirror(img, gt)
        if config.train_scale_array is not None:
			 img, gt, scale = random_scale(img, gt, config.train_scale_array)
        img = normalize(img, self.img_mean, self.img_std)
		# pytorch input need image value range from 0-1
        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)
		# get legal crop start position
        p_img , _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        p_gt , _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
		 #gt padding value is 225, image padding value is 0
        # add padding to those image with smaller shape
        return p_img, p_gt

# original size image input
class Seed_ValPre(object):
    def __init__(self, img_mean, img_std, target_size):
        self.img_mean = img_mean
        self.img_std  = img_std
        self.target_size = target_size
    def __call__ (self, img, gt):
        img_ = normalize(img, self.img_mean, self.img_std)
        return img_, gt


class Seed_TestPre(object):
    def __init__(self, img_mean, img_std, target_size):
        self.img_mean = img_mean
        self.img_std  = img_std
        self.target_size = target_size

    def __call__ (self, img, gt):
        # can be not 320 when evaluate
        img = normalize(img, self.img_mean, self.img_std)
        
        im_original_shape = img.shape[:2]   # [500 375]
        print(im_original_shape)
        
        img = mic.imresize(img, self.target_size/ float(max(im_original_shape)))

        resized_shape = img.shape[:2]   # 320 240
        margin  = [(self.target_size - resized_shape[0])//2, (self.target_size - resized_shape[1])//2 ]   # 0, 40
        img = cv2.copyMakeBorder(img, margin[0], self.target_size - resized_shape[0] - margin[0], 
            margin[1], self.target_size - resized_shape[1] - margin[1], cv2.BORDER_REFLECT_101)
        
        assert (img.shape[0] == img.shape[1] == self.target_size)
        return img, gt, dict(margin = margin, resized_shape = resized_shape, im_original_shape = im_original_shape)

class Sal_TrainPre(object):
    def __init__ (self, img_mean, img_std, target_size):
        self.img_mean = img_mean
        self.img_std  = img_std
        self.target_size = target_size
    def __call__(self, img, gt, sal):
        img, gt, sal = random_mirror(img, gt, sal)
        # plt.subplot(121)   plt.imshow(img)   plt.subplot(122) plt.imshow(gt)  plt.show()
        if config.train_scale_array is not None:
             img, gt,sal,  scale = random_scale(img, gt,sal, config.train_scale_array)
        img = normalize(img, self.img_mean, self.img_std)
        # pytorch input need image value range from 0-1
        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)
        # get legal crop start position
        p_img , _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        p_gt , _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_sal , _ = random_crop_pad_to_shape(sal, crop_pos, crop_size, 0)
        #gt padding value is 225, image padding value is 0
        # add padding to those image with smaller shape after scale
        return p_img, p_gt, p_sal

# original size image input and only do nomalized operator
class Sal_ValPre(object):
    def __init__(self, img_mean, img_std, target_size):
        self.img_mean = img_mean
        self.img_std  = img_std
        self.target_size = target_size
    def __call__ (self, img, gt, sal):
        img_ = normalize(img, self.img_mean, self.img_std)
        return img_, gt, sal

if __name__ == '__main__':
    img = Image.open("/home/ljh/桌面/VOCdevkit/VOC2012/JPEGImages/2008_005747.jpg")
    img_arr = np.array(img)
    gt = Image.open("/home/ljh/桌面/VOCdevkit/VOC2012/SegmentationClass/2008_005747.png").convert(mode = 'P')
    gt_arr = np.array(gt)
    preprocess = pre_train(para)
    img, gt = preprocess(img_arr, gt_arr, para)
