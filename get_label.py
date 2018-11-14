#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 11:19:53 2018

@author: hhhhhhhhhh
"""
import xml.dom.minidom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from config import config
xml_dir = '/home/hhhhhhhhhh/desktop/dataset/VOCdevkit/VOC2012/Annotations/'
gt_dir = '/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/SegmentationClass/'
img_dir = '/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/JPEGImages/'
# 21
def get_class_names(*args):
    return ['background', 'aeroplane', 'bicycle', 'bird','boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
            'person','pottedplant','sheep', 'sofa', 'train', 
            'tv/monitor']   
def get_class_colors(*args):
    return [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128],[128, 0, 0],
            [128, 0, 128], [128, 128, 0], [128, 128, 128],[0, 0, 64], [0, 0, 192],
            [0, 128, 64],[0, 128, 192], [128, 0, 64], [128, 0, 192], [128, 128, 64],
            [128, 128, 192], [0, 64, 0], [0, 64, 128],[0, 192, 0],[0, 192, 128], 
            [128, 64, 0], ]   # bgr, and cv read order

# no bg  
def read_gt(file_name):
    gt_path = gt_dir + file_name + '.png'
    gt = np.array(Image.open(gt_path).convert(mode = 'P'), dtype = np.uint8)   # h, w, c   # r, g, b 
    class_list = get_class_names()
    object_list = []
    file_class_idx = np.unique(gt)
    for class_idx in file_class_idx:  # other than 0 for bg, and 255 for ignored
        if class_idx != 0 and class_idx != 255:
            object_list.append(class_list[class_idx])
    return object_list
  
# return single .xml file object list , e.g ['cat', 'car'] , not consistence with gt seg
# a little unstrict about read .xml, may have other 'name'tag target, but canbe removed
def read_xml(file_name):
    xml_path = xml_dir +file_name + '.xml'
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    name_tag_list = root.getElementsByTagName('name')
    object_list =[]
    class_names = get_class_names()
    for tag in name_tag_list:
        if tag.firstChild.data not in object_list:
            # make sure no repeated object and leave out illegal object
            if tag.firstChild.data in class_names:
                object_list.append(str(tag.firstChild.data))
    return object_list

# turn single object list to a vector numpy, e.g ['cat', 'car'] -> (0, .. , 1, 0, 1, 0)
# object_list  no bg
def object2numpy(object_list):
    label_vector = np.zeros(config.num_classes)
#    label_vector[0] = 1   # for bg
    voc_class = get_class_names()
    for object in object_list:
        idx = voc_class.index(object)   # minus 1 for bg index
        label_vector[idx-1] = 1
    return label_vector

# get batch size .xml file gt vector, input batch file, output label vector, (batch_size, 21)
# batch_files got no  postfix, need to add .xml
# return np.array , need to turn to tensor, and variable
# batch_file should be a list not a single string
# get label from gt mask
def get_label_batch(batch_files):  
    batch_label = np.zeros([len(batch_files), config.num_classes])
    for batch_idx, batch_file in enumerate(batch_files):
        object_list = read_gt(batch_file)
        label_vector = object2numpy(object_list)
        batch_label[batch_idx] = label_vector
    return batch_label    
 
    

if __name__ == "__main__":
    batch_files = [ '2010_001960','2008_001787', 
     '2009_003867', '2008_003531', '2010_005008', '2009_000658','2010_004139' , 
     '2011_000850']
    batch_label = get_label_batch(batch_files)
    print(batch_label)
  
    
    
    
    
    
    
    
    
    
    
