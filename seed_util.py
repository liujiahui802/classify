# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
reload(sys)  
sys.setdefaultencoding('utf8')   # about python2.7 encoding problem whtch not in python3
import numpy as np
import cv2
import os.path as osp
from PIL import Image

from config import config
from update import compute_cc
###########################################################   get mask fg bg score for each image mask pixel  ############################

def get_conf( up_total_heatmaps , squeezed_map , gt_cls):
    total_heatmaps = np.zeros(up_total_heatmaps.shape) 
    total_heatmaps = up_total_heatmaps
    h, w = squeezed_map.shape
    conf = np.zeros((2, h,w))
    for h_i in range(h):
        for w_i in range(w):
            cls = int(squeezed_map[h_i][w_i])
            if cls != 0 :  # for fg pixel fg and bg score
#                print(cls-1)
                conf[0][h_i][w_i] = total_heatmaps[cls-1][h_i][w_i]   # for fg scores
                total_heatmaps[cls-1][h_i][w_i] = 0    # set to 0 and max has no effect 
                conf[1][h_i][w_i] = np.max(total_heatmaps[:,h_i,w_i])   # for bg scores
            else:  # for bg pixel fg  and bg scores
                conf[0][h_i][w_i] = np.max(total_heatmaps[np.array(gt_cls)-1 , h_i , w_i])
                total_heatmaps[np.array(gt_cls)-1 , h_i , w_i] = 0
                conf[1][h_i][w_i] = np.max(total_heatmaps[:,h_i,w_i])
                
    return conf

##############################################################  20-1 seed ############################################################################   
# 20-1 seed 
'''
input : 
        img :      batch * 3 * 320 * 320
        output:    1024 * 40 * 40
        fc_weight: training parameters in last GAP-FC layer

output : 
        up_total_heatmaps : 1024->21  upsampled real value heatmaps score
        confidence :        21 -> 1   max value score
'''
def _20_1up_heatmaps(img, output, fc_weight) :
    total_heatmaps = np.zeros((config.num_classes , output.size(1), output.size(2)))
    up_total_heatmaps = np.zeros((config.num_classes, img.size(2), img.size(3)))
    mux = []
    for class_idx in range(config.num_classes):
        for scoremap_idx in range(output.size(0)):
            each_score_maps = output[scoremap_idx].detach().cpu().numpy()
            total_heatmaps[class_idx] = total_heatmaps[class_idx] + each_score_maps * fc_weight[class_idx][scoremap_idx]
        #total_heatmaps[class_idx] = total_heatmaps[class_idx]  + fc_bias[class_idx]
        # upsample
        up_total_heatmaps[class_idx] = cv2.resize(total_heatmaps[class_idx], (img.size(3), img.size(2)))
        # normalize
        up_max = up_total_heatmaps[class_idx].max()
        mux.append(up_max)
        if up_max <= 0:
            up_max = 1
        up_total_heatmaps[class_idx] = up_total_heatmaps[class_idx] / up_max
    confidence = up_total_heatmaps.max(0)
    return up_total_heatmaps, confidence    # this is the normalized up_total heatmaps
  
  
# threshold lower than 0.2 to bg and leave out label not in image
# 0 for bg and 1-21 for 20classes
def _20_1threshold(threshold, squeezed_map, label, confidence):
    for h_idx in range(squeezed_map.shape[0]):
        for w_idx in range(squeezed_map.shape[1]):
            pix_class = squeezed_map[h_idx][w_idx]  # 1-20 class
            if (confidence[h_idx][w_idx]) < threshold:
                squeezed_map[h_idx][w_idx] = 0
            if label[pix_class-1] == 0 :
                squeezed_map[h_idx][w_idx] = 0
    return squeezed_map
    
##############################################################  new cls-1 seed ############################################################################   
# cls - 1
'''
input : 
        img :      batch * 3 * 320 * 320  (tensor)
        output:    1024 * 40 * 40         (tensor)
        fc_weight: training parameters in last GAP-FC layer
        
output :      
        up_class_heatmaps  :   cls heatmaps
        up_all_heatmaps    :   all heatmaps (for visualise , not calculate in squeezed map)
        cls_max_permap     :   max score value in each up_class_heatmaps
        all_max_permap     :   max score value in each up_all_heatmaps
        
''' 
def _cls_1up_heatmaps(img, output, fc_weight, label) :
    total_heatmaps = np.zeros((config.num_classes , output.size(1), output.size(2)))
    up_total_heatmaps = np.zeros((config.num_classes, img.size(2), img.size(3)))
    
    class_id = np.where(label ==1)[0]
    up_class_heatmaps = np.zeros((len(class_id), img.size(2), img.size(3)))
    
    for class_idx in range(config.num_classes):
        for scoremap_idx in range(output.size(0)):
            each_score_maps = output[scoremap_idx].detach().cpu().numpy()
            total_heatmaps[class_idx] = total_heatmaps[class_idx] + each_score_maps * fc_weight[class_idx][scoremap_idx]
        #total_heatmaps[class_idx] = total_heatmaps[class_idx]  + fc_bias[class_idx]
        # upsample
        up_total_heatmaps[class_idx] = cv2.resize(total_heatmaps[class_idx], (img.size(3), img.size(2)))
    for idx , pos in enumerate(class_id):
        up_class_heatmaps[idx] = up_total_heatmaps[pos]
        
    cls_max_permap = up_class_heatmaps.max(1).max(1)
    all_max_permap = up_total_heatmaps.max(1).max(1)
    return up_class_heatmaps, cls_max_permap , up_total_heatmaps, all_max_permap
    
    
##############################################################  squeezed  algorithm ############################################################################     
# squeeze upsampled heatmap to one map , which is the seed and turn class 0-19  to class 0-20
'''
input :  up_class_heatmops  :    can be 20 or cls
         maxx               :    max value in each map
         idx_label          :    gt label in this image
         threshold          :    squeezed threshold , when a pixel meets conflict, chose larger one before normalization

output : squeezed_map       :    seed map , 0-21 each label represents a class
'''
# 
def squeeze2( up_class_heatmaps, maxx,  idx_label, threshold = 0.5):
    cls_number = np.where(idx_label == 1)[0]
    
    squeezed_map = np.zeros((up_class_heatmaps.shape[1] , up_class_heatmaps.shape[2]))
    # get normalized up_heatmaps
    normalized_maps = np.zeros(up_class_heatmaps.shape)
    for idx in range(up_class_heatmaps.shape[0]):
        normalized_maps[idx, :, :] = up_class_heatmaps[idx] / maxx[idx]
    # pixel target squeeze
    for h_idx in range(squeezed_map.shape[0]):
        for w_idx in range(squeezed_map.shape[1]):
            cls_scores = normalized_maps[:, h_idx, w_idx]
            valid = np.where(cls_scores >= threshold)[0]
            valid_num = len(valid)
            if valid_num == 0 :
                squeezed_map[h_idx][w_idx] = 0                             # 0 for bg
            elif valid_num == 1:
                squeezed_map[h_idx][w_idx] = cls_number[valid[0]] + 1    # 0-19 t0 1-20
            elif valid_num >= 2:                                         # whene a pixel conflct, trace back to up_maps to chose the bigger value before normalization
                cls = np.argmax(up_class_heatmaps[:,h_idx,w_idx])
                squeezed_map[h_idx][w_idx] = cls_number[cls] + 1

    return squeezed_map
   

def squeeze3( up_total_heatmaps , maxx , idx_label, low_threshold = 0.5 , high_threshold = 0.75):
    cls_number = np.where(idx_label == 1)[0]   #  class order, 0-19
    squeezed_map = np.zeros((up_total_heatmaps.shape[1] , up_total_heatmaps.shape[2]))
    #get normalized up_heatmaps
    normalized_maps = np.zeros(up_total_heatmaps.shape)
    for idx in cls_number:
        normalized_maps[idx , : , :] = up_total_heatmaps[idx] / maxx[idx]   # only normalize valid class maps. other maps set to 0
        
    # rank maxx per map , thresould top5 valid class and other valid class according to diffenrent threshold
    rank_max_pos = np.argsort(maxx)[::-1]   # max to min map position
    top5 = rank_max_pos[0:5]
    for cls in cls_number:
        if cls in top5:
            threshold = low_threshold
        else:
            threshold = high_threshold
        normalized_maps[cls][normalized_maps[cls]  < threshold ] = 0    #  valid calss normalized maps value under threshold is set to 0
    
    # squeezed top5 valid and not top5 valid classes , use other class maps to remove fuzzy area
    for h_idx in range(squeezed_map.shape[0]):
        for w_idx in range(squeezed_map.shape[1]):
            cls_scores = up_total_heatmaps[:, h_idx, w_idx]    # compare value , not normalized value
            # first compare the largest score one in all classes
            pix_cls = np.argmax(cls_scores)
            if pix_cls in cls_number:
                if normalized_maps[pix_cls][h_idx][w_idx] > 0:
                    # assure in valid class's valid area
                    squeezed_map[h_idx][w_idx] = pix_cls + 1
    return squeezed_map



def squeeze4(up_class_heatmaps , maxx , idx_label, threshold = 0.75):
    cls_number = np.where(idx_label == 1)[0]
    squeezed_map = np.zeros((up_class_heatmaps.shape[1] , up_class_heatmaps.shape[2]))
    
    min_cls_value = np.min(maxx * threshold )  # cause chose minimum value, threshold should be larger
    
    for h_idx in range(squeezed_map.shape[0]):
        for w_idx in range(squeezed_map.shape[1]):
            cls_scores = up_class_heatmaps[: , h_idx, w_idx]
            pixel_cls = cls_number[np.argmax(cls_scores)] + 1
            pixel_max_value = np.max(cls_scores)
            if pixel_max_value >= min_cls_value:
                squeezed_map[h_idx][w_idx] = pixel_cls
    return squeezed_map
    
    
def squeeze5(up_class_heatmaps , maxx, idx_label , threshold = 0.2):
    cls_number = np.where(idx_label == 1)[0]
    squeezed_map = np.zeros((up_class_heatmaps.shape[1] , up_class_heatmaps.shape[2]))
    
    normalized_maps = np.zeros(up_class_heatmaps.shape)
    for idx in range(up_class_heatmaps.shape[0]):
        normalized_maps[idx, :, :] = up_class_heatmaps[idx] / maxx[idx]
   
    for h_idx in range(squeezed_map.shape[0]):
        for w_idx in range(squeezed_map.shape[1]):
            cls_scores = up_class_heatmaps[:, h_idx , w_idx]
            max_cls = np.argmax(cls_scores)
            if normalized_maps[max_cls , h_idx , w_idx] > threshold:
                squeezed_map[h_idx][w_idx] = cls_number[max_cls] + 1   
#    squeezed_map = cls_number[up_class_heatmaps.argmax(axis = 0)]
#    confidence = up_class_heatmaps.max(0)
#    seed = squeezed_map * (confidence > threshold)
    
    return squeezed_map        
 
# select seed5 area with seed3 high confidence area
def intersec_seed_35(seed3 , seed5):
    seed5_cc_list , seed5_cc_size = compute_cc(seed5!=0 , minarea = 0)
    for i in range(len(seed5_cc_list)):
        inter_num = np.sum(np.logical_and(seed5_cc_list[i]!=0 , seed3!=0))
        if inter_num==0: # remove for no intersection with seed3 condition
            seed5[seed5_cc_list[i] == 1] = 0
    return seed5
            

# to change score according to cls
def squeeze6(all_max_permap , cls_max_permap , up_class_heatmaps , gt_cls , threshold = 0.2):
    squeezed_map = np.zeros((up_class_heatmaps.shape[1] , up_class_heatmaps.shape[2]))
    maxx = np.max(all_max_permap)
# multiple by maxx/ per max
    for idx in range(up_class_heatmaps.shape[0]):
        up_class_heatmaps[idx, : , :] = up_class_heatmaps[idx , : , :] *float(maxx / cls_max_permap[idx])
    
    
    normalized_maps = np.zeros(up_class_heatmaps.shape)
    for idx in range(up_class_heatmaps.shape[0]):
        normalized_maps[idx, :, :] = up_class_heatmaps[idx] / cls_max_permap[idx]
   
    for h_idx in range(squeezed_map.shape[0]):
        for w_idx in range(squeezed_map.shape[1]):
            cls_scores = up_class_heatmaps[:, h_idx , w_idx]
            max_cls = np.argmax(cls_scores)
            if normalized_maps[max_cls , h_idx , w_idx] > threshold:
                squeezed_map[h_idx][w_idx] = gt_cls[max_cls]
    
    
    return squeezed_map
    
def squeeze7(up_class_heatmaps , gt_cls):
    squeezed_map = np.zeros((up_class_heatmaps.shape[1] , up_class_heatmaps.shape[2]))
    print(up_class_heatmaps.shape)
    squeezed_map = np.argmax(up_class_heatmaps ,axis = 0)
    print(squeezed_map.shape)
    for x in range(squeezed_map.shape[0]):
        for y in range(squeezed_map.shape[1]):
            squeezed_map[x][y] = gt_cls[squeezed_map[x][y]]
    
    return squeezed_map

def squeeze4exp(heatmaps , threshold = 0.2):   # 20 for heatmaps
    squeezed_map = np.zeros((heatmaps.shape[1] , heatmaps.shape[2]))
    
    max_per = heatmaps.max(1).max(1)
    heatmaps_nor = np.zeros(heatmaps.shape)
    for idx in range(heatmaps.shape[0]):
        heatmaps_nor[idx, :, :] = heatmaps[idx , : , :] / float(max_per[idx])
        
    for h_idx in range(heatmaps.shape[1]):
        for w_idx in range(heatmaps.shape[2]):
            cls_scores = heatmaps[:, h_idx , w_idx]
            max_cls = np.argmax(cls_scores)
            if heatmaps_nor[max_cls , h_idx , w_idx] > threshold:
                squeezed_map[h_idx][w_idx] = max_cls + 1
    return squeezed_map

   
##############################################################  read sal ############################################################################     
def read_initiaL_sal(file_name):
    file_name = file_name + '.jpg'
    data = np.array(Image.open(osp.join('/home/hhhhhhhhhh/desktop/project_seg/new_saliency/test_output',file_name)), dtype = np.uint8) 
    h , w = data.shape
    
    for h_idx in range(h):
    	for w_idx in range(w):
    		if data[h_idx][w_idx] >= 100:
    			data[h_idx][w_idx] = 1
    		else:
    			data[h_idx][w_idx] = 0
    #print(data.max(0).max(0))
    return data
    
##############################################################  initial metric ############################################################################   
# add all the history image together
def get_num4TFPN(pred_real_all_classes, idx_label , gt, squeezed_map):
    h,  w = np.shape(gt)
    overlay = np.zeros((2, h, w))
    overlay[0] = gt
    overlay[1] = squeezed_map
    label = [0]  # 0-20 etc.
    for class_idx, one in enumerate(idx_label):
        if one ==1:
            label.append(class_idx+1)  
    for class_ in label:
        for h_idx in range(h):
            for w_idx in range(w):
                gt = int(overlay[0][h_idx][w_idx])    # class index for  0-20
                seed = int(overlay[1][h_idx][w_idx])
                if gt == class_ and seed == class_ :
                    pred_real_all_classes[class_][0] = pred_real_all_classes[class_][0] +1
                elif gt != class_ and seed != class_ :
                    pred_real_all_classes[class_][1] = pred_real_all_classes[class_][1] +1   
                elif gt != class_ and seed == class_ :
                    pred_real_all_classes[class_][2] = pred_real_all_classes[class_][2] +1
                elif gt == class_ and seed != class_ :
                    pred_real_all_classes[class_][3] = pred_real_all_classes[class_][3] +1
    return pred_real_all_classes

def get_metric(pred_real_all_classes):
    precision = np.zeros(config.num_classes +1)
    recall = np.zeros(config.num_classes +1)
    for class_idx in range(config.num_classes+1) :
        if 0 not in pred_real_all_classes[class_idx]:
            precision[class_idx] = float(pred_real_all_classes[class_idx][0]) / float(pred_real_all_classes[class_idx][0] + pred_real_all_classes[class_idx][2]) 
            recall[class_idx] = float(pred_real_all_classes[class_idx][0]) / float(pred_real_all_classes[class_idx][0] + pred_real_all_classes[class_idx][3]) 
    return precision , recall
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    