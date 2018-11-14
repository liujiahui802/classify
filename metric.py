#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 09:58:03 2018

@author: hhhhhhhhhh
"""
from sklearn.metrics import precision_score,  recall_score,  classification_report,  f1_score
import numpy as np

from lib.datasets.voc.voc import Seed_VOC
# sal measure
def F_measure(b, precision, recall):
    F_measure = ((1+b**2)*precision*recall)/(b*b*precision*recall)
    return F_measure

#average absolute pielwise difference between two saliency maps
def MAE(pred, gt):
    h, w = pred.shape
    dif = 0
    for h_idx in range(h):
        for w_idx in range(w):
            dif_pixel = abs(pred[h_idx][w_idx] - gt[h_idx][w_idx])
            dif = dif + dif_pixel
    avg_dif = dif/(h*w)
    return avg_dif

# compute mean iou and class iou, including bg, 21 classes in total
# has leave out 255 ignore in label , and not allowed 255 in pred 
'''
input  : pred_vector(flattened from mask map) , gt_vector
output : mean_IoU, class_IoU array
'''
# label has bg , including 0 and 255, but some picture may not including bg
def compute_mean_iou(pred, label , has_ignore = False):
    unique_labels = (np.unique(label)).tolist() 
    # record 255 pixel position in labels and leave out when counting U
    if 255 in unique_labels:
        has_ignore = True
        ignore = np.where(label ==255)
        unique_labels.remove(255)

    # for calculating all_cls iou 
    I = np.zeros(21)
    U = np.ones(21)

    for  cls in unique_labels:
        pred_i = pred == cls
        label_i = label == cls

        I[cls] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[cls] = float(np.sum(np.logical_or(label_i, pred_i)))

    # remove ignore pixels in counting U
    if has_ignore :
        for ignore_pixel in ignore[0]:
            pred_label = pred[ignore_pixel]
            if pred_label in unique_labels:
                U[pred_label] =  U[pred_label] - 1

    class_iou = I / U
    
    fg_mean_iou = 0
    for label in unique_labels:
        if label != 0:
            fg_mean_iou = fg_mean_iou + class_iou[label]
    # for case not including bg, calculate its fg iou
    if 0 in unique_labels:
        fg_mean_iou = fg_mean_iou / (len(unique_labels)-1)
        bg_iou = class_iou[0]
    else:
        fg_mean_iou = fg_mean_iou / len(unique_labels)
        bg_iou = 0
      
    return fg_mean_iou, bg_iou , class_iou    # has peoblem in class iou

# Compute the average segmentation accuracy across all classes
'''
input  : pred and label vector
output : global_accuracy
'''
def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
# If there are no pixels from a certain class in the GT, 
# it returns NAN because of divide by zero
# Replace the nans with a 1.0.
'''
input  : pred and label vector
output : class_accuracy_vector
'''
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])
    return accuracies


# evaluate single flattened pred and gt, class_num is 21 and make metric report
'''
input : pred_map, gt_map
output : classfication_report(precision, recall, f1) and iou
'''
def evaluate_single_segmentation(pred, label, num_classes, score_averaging="weighted"):

    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    fg_iou, bg_iou , class_iou = compute_mean_iou(flat_pred, flat_label)
    
    #print(classification_report(flat_label, flat_pred, target_names=Seed_VOC.get_class_names()))
    return global_accuracy, class_accuracies, prec, rec, f1, fg_iou, bg_iou , class_iou

if __name__ == '__main__':
    label = np.array([[1,2,255, 3,2,1],[2,255,3 , 1,2,1]])
    pred = np.array([[1,255,1,0,0,1], [2,2,3,1,0,0]])
    

    fg_iou, bg_iou , cls_iou = compute_mean_iou(pred.flatten(), label.flatten())