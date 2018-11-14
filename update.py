#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# this file is for all seed and sal update algorithm
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from random import choice
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian
from skimage import measure
from skimage import morphology
import scipy.ndimage as nd
import os.path as osp
from PIL import Image
from util import make_palette


from slic import  SLICProcessor
from metric import MAE

# combination of seed5 and seed0.8 threshold results , when having interction , do logic_or operation
def comb_seed(img_path1 , img_path2 ):
 

        seed1 = np.array(Image.open(img_path1).convert(mode = 'P'), dtype = np.uint8)
        seed2 = np.array(Image.open(img_path2).convert(mode = 'P') , dtype = np.uint8)
        seg = np.zeros(seed1.shape)
        
        gtc1 = np.unique(seed1).tolist()
        gtc1.remove(0)
        gtc2 = np.unique(seed2).tolist()
        gtc2.remove(0)
        
        seed1_cls_cc = {}
        seed2_cls_cc = {}
        seed1_cls_cc_size = {}
        seed2_cls_cc_size = {}
        seed1_cls_cc_inter = {}
        seed2_cls_cc_inter = {}
        # for seed1 connected components
        for c in gtc1:
            seed1_cls_cc[c] , seed1_cls_cc_size[c] = compute_cc(seed1 == c , minarea = 20)             # , superpixel = p)
            seed1_cls_cc_inter[c] = np.zeros(len(seed1_cls_cc[c]))
            
            # juedge intersection in each class seed cc
            for idx , c1 in enumerate(seed1_cls_cc[c]):
                inter_num =np.sum(np.logical_and(c1 , seed2==c ))
                if inter_num >= 20:
                    seed1_cls_cc_inter[c][idx] = 1
        # for seed2 connected components        
        for c in gtc2:
            seed2_cls_cc[c] , seed2_cls_cc_size[c] = compute_cc(seed2 == c , minarea = 20)             # , superpixel = p)
            seed2_cls_cc_inter[c] = np.zeros(len(seed2_cls_cc[c]))
            
            # to judge intersection
            for idx , c2 in enumerate(seed2_cls_cc[c]):
                inter_num =np.sum(np.logical_and(c2 , seed1==c ))
                if inter_num >= 20:
                    seed2_cls_cc_inter[c][idx] = 1
                    
        # overwite seg according to 2 seed and intersection flag
        for c in gtc1  :
            for idx,  flag in enumerate(seed1_cls_cc_inter[c]):
                if flag == 1:
                    seg[seed1_cls_cc[c][idx] == 1] = c
        for c in gtc2  :
            for idx,  flag in enumerate(seed2_cls_cc_inter[c]):
                if flag == 1:
                    seg[seed2_cls_cc[c][idx] == 1] = c
                    

        
        return seg
            
        
    
######################################################################################################################################
########################################################################################################################################
# sal 2 value crf
def dense_crf(img, output_probs):
    h , w = output_probs.shape
    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
    
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U.astype(np.float32))
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img.astype(np.ubyte), compat=10)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q

# seed multi-label crf
def seed_dense_crf(img, labels):
    print('using ---------------------  post crf')
    h , w = labels.shape
    #n_labels = len(np.unique(labels))
    n_labels = 21
    # Example using the DenseCRF2D code
    d = dcrf.DenseCRF2D(w , h , n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only .
    d.addPairwiseGaussian(sxy=3, compat=10, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=img,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(10)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))
    
    return Q
    
# this for combination of sal and seed when multi seeds exist, use crf to modify the edge of multiple classes
def CRF(image, unary, crf_param, scale_factor=1.0, maxiter=10):
    assert (image.shape[:2] == unary.shape[:2])
    scale_factor = float(scale_factor)
    if 'seed' in crf_param:
        bi_w = 10.
        bi_x_std = 80.
        bi_r_std = 13.
    elif 'deeplab' in crf_param:
        bi_w = 4.
        bi_x_std = 121.
        bi_r_std = 5.
    elif 'custom' in crf_param:
        _custom, bi_w_, bi_x_std_, bi_r_std_ = crf_param.split('-')
        bi_w = float(bi_w_)
        bi_x_std = float(bi_x_std_)
        bi_r_std = float(bi_r_std_)
    else:
        raise NotImplementedError
    pos_w = pos_x_std = 3.
    H, W = image.shape[:2]
    nlabels = unary.shape[2]
    # initialize CRF
    # crf = DenseCRF(W, H, nlables)
    crf = dcrf.DenseCRF2D(W, H, nlabels)
    # set unary potentials
    # crf.set_unary_energy(-unary.ravel().astype('float32'))
    crf.setUnaryEnergy(-unary.transpose((2, 0, 1)).reshape((nlabels, -1)).copy(order='C').astype('float32'))
    # set pairwise potentials
    w1 = int(bi_w)
    theta_alpha_1 = int(bi_x_std / scale_factor)    
    theta_alpha_2 = int(bi_x_std / scale_factor)
    theta_beta_1 = int(bi_r_std)
    theta_beta_2 = int(bi_r_std)
    theta_beta_3 = int(bi_r_std)
    w2 = int(pos_w)
    theta_gamma_1 = int(pos_x_std / scale_factor)   
    theta_gamma_2 = int(pos_x_std / scale_factor)
#    crf.addPairwiseEnergy(w1,
#                             theta_alpha_1,
#                             theta_alpha_2,
#                             theta_beta_1,
#                             theta_beta_2,
#                             theta_beta_3,
#                             w2,
#                             theta_gamma_1,
#                             theta_gamma_2,
#                             image.ravel().astype('ubyte'))
    # same as pydensecrf demo usage
    crf.addPairwiseGaussian(sxy=(theta_gamma_1, theta_gamma_2),
                            compat=w2,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)   # 3,3,3
    crf.addPairwiseBilateral(sxy=(theta_alpha_1, theta_alpha_2), srgb=(theta_beta_1, theta_beta_2, theta_beta_3),
                             rgbim=np.ascontiguousarray(image), compat=w1, kernel=dcrf.DIAG_KERNEL,
                             normalization=dcrf.NORMALIZE_SYMMETRIC)   # 121,121   5 5 5   4 
    # run inference
    prediction = np.array(crf.inference(maxiter)).reshape((nlabels, H, W)).transpose((1, 2, 0))
    return prediction

def mask_inference(image_original, gap_seed, ov_gap):
    involved_gtc = np.unique(np.array([o[0] for o in ov_gap]))
    unary = np.zeros((image_original.shape[0], image_original.shape[1], len(involved_gtc)), dtype=np.float)   # shape of image
    for ov_tmp in ov_gap:
        gtc, gc = ov_tmp
        heatmap = (gap_seed == gtc).astype(np.float32)
        heatmap = np.maximum(heatmap, 1e-3)
        heatmap = np.minimum(heatmap, 1 - 1e-3)                    # heatmap score from 0.001 to 1 - 0.001
        heatmap = nd.filters.gaussian_filter(heatmap, 10)          #  guassian fuzzy filter
        unary[:, :, np.where(gtc == involved_gtc)[0][0]] = heatmap
    unary_smoothed = CRF(image_original, unary, crf_param='deeplab')

    return involved_gtc[unary_smoothed.argmax(2)]

#######################################################################################################################
# CLS_crf + seed intercetion chose better boundary and supress unreasonable area in sal 
# ignore unreasonable area as bg or other sal intersection class
def CLS_seed_chose( mask , seed ,up_class_heatmaps , gt_cls, ignore = True):
    print('=================  in  CLS seed chose step ================')
    print(gt_cls)
    # get mask gtc cc
    mask_cls_cc = {}
    mask_cls_cc_size = {}
    for gtc in gt_cls:
        mask_cls_cc[gtc] , mask_cls_cc_size[gtc] = compute_cc(mask == gtc , minarea = 20)
    
    for gtc in gt_cls:
        for mc in mask_cls_cc[gtc]:
            inter_num = np.sum(np.logical_and(mc , seed == gtc))
            if inter_num < 20:   # for unreasonable area, process it
                if ignore == True:
                    print('-----------------  ignore unreasonable area-------------')
                    mask[mc == 1] = 0
                else:
                    around_cls = Around_cls(mask , mc)
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  around cls')
                    print(around_cls)
                    print('unreasonable area gtc----')
                    print(gtc)
                    if around_cls == []:
                        print('around = [] ')
                        mask[mc == 1] = 0
# no order assign                    
                    elif len(around_cls) == 1:
                        print(' around = 1')
                        mask[mc ==1] = around_cls[0]
                    elif len(around_cls) >1:     # chose one of other valid class in around according to scoremaps
                        print(' aroudn = 2')
                        around_valid_heatmaps = np.zeros((len(around_cls) , up_class_heatmaps.shape[1] , up_class_heatmaps.shape[2]))
                        for idx in range(len(around_cls)):
                            around_valid_heatmaps[idx] = up_class_heatmaps[gt_cls.index(around_cls[idx])]
                        x , y = np.where(mc == 1)
                        for i in range(len(x)):
                            mask[x[i]][y[i]] = around_cls[np.argmax(around_valid_heatmaps[: , x[i] , y[i]])]
    
# all gt_cls score assign  
#                    elif around_cls != []:  # means has surrounding area in sal ,  but chose second argmax in all gt_cls
#                        print('has around-=-=-=-=-=-=')
#                        # construct other gt_cls heatmaps
#                        this_cls_index = gt_cls.index(gtc)
#                        gt_other_valid_heatmaps = np.zeros((len(gt_cls)-1 , up_class_heatmaps.shape[1] , up_class_heatmaps.shape[2]))
#                        for idx in range(len(gt_cls)-1):
#                            idx_gt_cls = idx
#                            if idx >= this_cls_index:
#                                idx_gt_cls = idx +1
#                            gt_other_valid_heatmaps[idx] = up_class_heatmaps[idx_gt_cls]
#                        gt_ = []
#                        for i in gt_cls:
#                            gt_.append(i)
#                        gt_.remove(gtc)
#                        
#                        # chose the second max cls and assign
#                        x, y = np.where(mc == 1)
#                        for i in range(len(x)):
#                            mask[x[i]][y[i]] = gt_[np.argmax(gt_other_valid_heatmaps[ : , x[i] , y[i]])]

    
    return mask
                        
                
def Around_cls(mask , mc):
    x, y = np.where(mc==1)
    other_valid_cls = []
    for i in range(len(x)):
        around_cls = pix_around_cls(x[i], y[i], mask) # around cls not have this mask_cc cls  , cause it don't intersection with seed
        for c in around_cls:
            if c not in other_valid_cls:
                other_valid_cls.append(c)
    
    return other_valid_cls
# get single pixel around cls different from this pixel class   
def pix_around_cls(x, y , mask):
    up = (max(x-1, 0) , y)
    down = (min(x + 1 , mask.shape[0]-1), y)
    left = (x, max(y-1 , 0))
    right = (x , min(y +1 , mask.shape[1] - 1))
    pix_cls =mask[x][y]  # this pix_class , which not have intersection with original seed gtc area
    around_cls = [pix_cls]
    if mask[up[0]][up[1]] not in around_cls:
        around_cls.append(mask[up[0]][up[1]])
    if mask[down[0]][down[1]] not in around_cls:
        around_cls.append(mask[down[0]][down[1]])
    if mask[left[0]][left[1]] not in around_cls:
        around_cls.append(mask[left[0]][left[1]])
    if mask[right[0]][right[1]] not in around_cls:
        around_cls.append(mask[right[0]][right[1]])
        
    around_cls.remove(pix_cls)
    if 0 in around_cls:
        around_cls.remove(0)
    return around_cls
    
######################################################################################################################
#combination algorithm
'''
input:   sal , seed : nd.array
         img        : original image
         gt_cls     : gt class order list , 1-20 , eg.[3,19] 
output:
         seg        : mask seg, 0-20 , 0 for bg , 1-20 for fg

'''
def sal_seed1(sal, seed , img ,up_class_heatmaps , maxx,  up_total_heatmaps,  gt_cls  ):                                            #     , p ):    
    imgshape = img.shape[1:3]
    seg = np.ones(imgshape , dtype = np.uint8) *255 
    gap_cls_cc = {}
    gap_cls_cc_size = {}
    gap_cls_cc_usage = {}
# get seed and sal cc, gtc has no bg , class is 1-21
    for gtc in gt_cls:
        gap_cls_cc[gtc] , gap_cls_cc_size[gtc] = compute_cc(seed == gtc , minarea = 0)             # , superpixel = p)
        gap_cls_cc_usage[gtc] = np.zeros(len(gap_cls_cc[gtc]))
    sal_cc , sal_size = compute_cc(sal)                                                        # , superpixel = p) 

# for sal algorithm
    for sc in sal_cc:
        ov_gap = []
        for gtc in gt_cls:
            for idx , gc in enumerate(gap_cls_cc[gtc]):
                ov = compute_iou(sc, gc)
                if ov > 0 :
                    if gtc ==0:
                        pass
                    else:
                        ov_gap.append([gtc, gc])    #  only append gap conponents has iou with saliency
#                    if ov > 0.1:                       #  leave out used gap cc in next step
                        gap_cls_cc_usage[gtc][idx] = 1   
        # to calculate how many seed classes for each one sal connected component  sc    
        if ov_gap == []:
            n_ovcls = 0
        else: 
            n_ovcls = len(np.unique(np.array([o[0] for o in ov_gap])))

        if n_ovcls == 0:  
            seg = set4sal_cc(seg, sc, new_value = 255)
        elif n_ovcls == 1:
            seg = set4sal_cc(seg , sc, new_value = ov_gap[0][0])
        elif n_ovcls >= 2:
#            print('using CRF uniary modify')
#            pred = mask_inference(img.transpose(1,2,0), seed, ov_gap)
#            seg = set_infer_pred( seg, pred, sc)
        
            print('using CLS2')
            seg = heat_multiseed( seg , sc, up_class_heatmaps , gt_cls)
            
            
# for seed algorithm  
    for gtc in gt_cls:
        for idx, gc in enumerate(gap_cls_cc[gtc]):
            if gap_cls_cc_usage[gtc][idx]:   # gt_cls 1-20
                if gtc == 0 :   #  ???? about bg class        gc other than intersection with sal area, set to 0
                    print('------ has gtc = 0 ----------')                    
                    seg[np.maximum(0, gc.astype(np.int) - (gc & sal).astype(np.int)).astype(np.bool).astype(np.int)] = 0
            else:  ############ reserve seed place ####  can change ############
                seg = set4sal_cc(seg , gc, new_value = gtc)   # reserve seed cc which has no intersectio with sal 
    #seg[seg == 255] = 0
    seg = set4sal_cc(seg, seg, old_value = 255, new_value = 0)
    
    # use all_up_heat_maps to remove sal useless area
    #seg = all_heat_remove( seg ,up_total_heatmaps )
    
    # when there is multi seed intersection , there is a cls _heat comparison , but when there is only one cls in 
        #  sal ,there is no comparison , to include this situation , use cls_heatmaps to remove 
#    seg = cls_heat_remove( seg, up_class_heatmaps , gt_cls)
  
  
   
    return seg

#combination algorithm
'''
input:   sal , seed : nd.array
         img        : original image
         gt_cls     : gt class order list , 1-20 , eg.[3,19] 
         nor_threshold : normaliezed interested area cls_map threshold
         int_threshold : sal_cc iou cls_heatmap all in threshold         
         
output:
         seg        : mask seg, 0-20 , 0 for bg , 1-20 for fg

'''
def sal_seed2(sal, seed, up_class_heatmaps , up_total_heatmaps, maxx , gt_cls , nor_threshold = 0.05 , int_threshold = 0.2):                                          
    imgshape = sal.shape
    seg = np.ones(imgshape , dtype = np.uint8) *255 
    gap_cls_cc = {}
    gap_cls_cc_size = {}
    gap_cls_cc_usage = {}
# get seed and sal cc, gtc has no bg , class is 1-21
    for gtc in gt_cls:
        gap_cls_cc[gtc] ,gap_cls_cc_size[gtc] = compute_cc(seed == gtc)                                  
        gap_cls_cc_usage[gtc] = np.zeros(len(gap_cls_cc[gtc]))
    sal_cc , sal_size = compute_cc(sal)                                                       

# get normalized up_class_heatmaps , and value under 0.2 is set to 0
    normalized_maps = np.zeros(up_class_heatmaps.shape)
    for idx in range(up_class_heatmaps.shape[0]):
        normalized_maps[idx , : , :] = up_class_heatmaps[idx] / maxx[idx]
        normalized_maps[idx][normalized_maps[idx] < nor_threshold ] = 0   # for normalized value under 0.2 , set to 0

# calculate sal area percentage of 0.2 up_class_heatmaps value , judge sal_cc interested area , all in ,or just the intersection 
# iof 0,2 heatmaps and sal_cc , return interested new_sal_cc
    for idxx , sc in enumerate(sal_cc):
        # to judge whether sal_cc pixel has interested heatmap value
        size_sc = sal_size[idxx] 
        interest_pixels = 0
        pos_x , pos_y = np.where(sc == 1)
        for idx in range(len(pos_x)):
            cls_normalized_values = normalized_maps[: , pos_x[idx] , pos_y[idx]]
            if len(np.where(cls_normalized_values >= nor_threshold)[0]) != 0:   # has interest 
                interest_pixels = interest_pixels + 1
        # to get new sal_cc according to int_threshold  
        # when > threshold , sal_cc all in ,not change
        if ( float(interest_pixels) / float(size_sc) ) < int_threshold :
            print('-------------------------------------------------------------------------------')
            print(interest_pixels)
            print(size_sc)
            print(float(interest_pixels) / float(size_sc))              
            print('--------------------------------------------------------------------has new_sal')
            for idx in range(len(pos_x)):
                cls_normalized_values = normalized_maps[: , pos_x[idx] , pos_y[idx]]
                if len(np.where(cls_normalized_values >= nor_threshold)[0]) == 0:     # has no any interest
                    sal_cc[idxx][pos_x[idx]][pos_y[idx]] = 0  

# for sal algorithm
    for sc in sal_cc:
        ov_gap = []
        for gtc in gt_cls:
            for idx , gc in enumerate(gap_cls_cc[gtc]):
                ov = compute_iou(sc, gc)
                if ov > 0 :
                    ov_gap.append([gtc, gc])    
                    if ov > 0.1:      # for those little intersection
                        gap_cls_cc_usage[gtc][idx] = 1  
        if ov_gap == []:
            n_ovcls = 0
        else: 
            n_ovcls = len(np.unique(np.array([o[0] for o in ov_gap])))
        #print(n_ovcls)
        if n_ovcls == 0:  
            seg = set4sal_cc(seg, sc, new_value = 255)
        elif n_ovcls == 1:
            seg = set4sal_cc(seg , sc, new_value = ov_gap[0][0])
        elif n_ovcls >= 2:#   when sal intersection with multiple seed, use class_heatmaps to chose larger value
            seg = heat_multiseed( seg , sc, up_class_heatmaps , gt_cls)
# for seed algorithm  
    for gtc in gt_cls:
        for idx, gc in enumerate(gap_cls_cc[gtc]):
            if gap_cls_cc_usage[gtc][idx]:
                if gtc == 0 :   #  ???? about bg class
                    seg[np.maximum(0, gc.astype(np.int) - (gc & sal).astype(np.int)).astype(np.bool).astype(np.int)] = 0
            else:  ###    ######### reserve seed place ####  can change ############
                seg = set4sal_cc(seg , gc, new_value = gtc)   # reserve seed cc which has no intersectio with sal 
    #seg[seg == 255] = 0
    seg = set4sal_cc(seg, seg, old_value = 255, new_value = 0)

    # use all_up_heat_maps to remove sal useless area
#    seg = all_heat_remove( seg ,up_total_heatmaps )

    # when there is multi seed intersection , thereis a cls _heat comparison , but when there is only one cls in 
        #  sal ,there is no comparison , to include this situation , use cls_heatmaps to remove 
#    seg = cls_heat_remove( seg, up_class_heatmaps , gt_cls)

    return seg



'''for MCG SAL SEED combination'''
def sal_seed_mcg1(sal, seed , proposals, img , up_class_heatmaps , up_total_heatmaps,  gt_cls  ):                                            #     , p ):    
    imgshape = img.shape[1:3]
    seg = np.ones(imgshape , dtype = np.uint8) *255 
    gap_cls_cc = {}
    gap_cls_cc_size = {}
    gap_cls_cc_usage = {}
# get seed and sal cc, gtc has no bg , class is 1-21
    for gtc in gt_cls:
        gap_cls_cc[gtc] ,gap_cls_cc_size[gtc] = compute_cc(seed == gtc , minarea = 100)             # , superpixel = p)
        gap_cls_cc_usage[gtc] = np.zeros(len(gap_cls_cc[gtc]))
    sal_cc , sal_size = compute_cc(sal)                                                        # , superpixel = p) 

# for calculate unused seed_cc, for each sal_cc , search through all the gtc seed_Cc
    ov_gap_allsal = []    
    for sc in sal_cc:
        ov_gap = []
        for gtc in gt_cls:
            for idx , gc in enumerate(gap_cls_cc[gtc]):
                ov = compute_iou(sc, gc)
                if ov > 0 :
                    if gtc ==0:
                        pass
                    else:
                        ov_gap.append([gtc, gc])    #  only append gap conponents has iou with saliency
                    if compute_rate(sc , gc) > 0.1:    # intersection  / seed_cc is larger than 0.1
                        gap_cls_cc_usage[gtc][idx] = 1   #  leave out used gap cc in next step
        ov_gap_allsal.append(ov_gap)
                        

#1
# for will be unused seed_cc , should have priority if prop assignment                     
    assign_prop_list = [] 
    for gtc in gt_cls:
        for idx, gc in enumerate(gap_cls_cc[gtc]):
            place_gc = np.where(gc == 1)
            if gap_cls_cc_usage[gtc][idx] == 0:
                likely_prop , prop_size = find_max_prop( gc , proposals ) 
                place_prop = np.where(likely_prop == 1)
                assign_prop_list.append([prop_size , likely_prop ,gtc])
    # has priority in prop size when assignment
    if assign_prop_list != []:
        print('-----------------processing 1 ---------------')        
        assign_prop_list.sort(key = takefirst)   # min to max prop_size
        assign_prop_list[::-1]     #max to min priority
        
        for prop in assign_prop_list: 
        # for unused seed and max intersection prop, set pixel in cls_heatmaps larger than other class
         #   seg[prop[1] == 1] = prop[2]
            seg = heat_unused_seed(seg, prop[1] , up_total_heatmaps, prop[2] , gt_cls)
        


#2
# for sal seed combination 
    print('1111111111111111111111111  processing 2  1111111111111111111')    
    for idx , sc in enumerate(sal_cc):
        ov_gap = ov_gap_allsal[idx]
# to calculate how many seed classes for each one sal connected component  sc    
        if ov_gap == []:
            n_ovcls = 0
        else: 
            n_ovcls = len(np.unique(np.array([o[0] for o in ov_gap])))

        if n_ovcls == 0:  
            seg = set4sal_cc(seg, sc, new_value = 255)
        elif n_ovcls == 1:
            seg = set4sal_cc(seg , sc, new_value = ov_gap[0][0])
        elif n_ovcls >= 2:
#            pred = mask_inference(img.transpose(1,2,0), seed, ov_gap)
#            seg = set_infer_pred( seg, pred, sc)
            seg = heat_multiseed( seg , sc, up_class_heatmaps , gt_cls)


# set for bg  
    seg = set4sal_cc(seg, seg, old_value = 255, new_value = 0)


# 3 for mask seg, use mcg to modify sal_seed mask    work !!! modify the results edge
    print('============  processing 3 ===========  ')    
    seg_cc , seg_size = compute_cc(seg != 0)    
    for segc in seg_cc:
        seg_prop ,seg_size = find_max_prop(segc , proposals)
        rate = compute_rate(seg_prop , segc)
        if rate > 0.75:
            seg = heat_multiseed ( seg , seg_prop , up_class_heatmaps , gt_cls)
   
    return seg

#################################################### functions ############################################################################
# find the max intersection / proposal rate proposal with arr, if no intersection prop , return prop is all 0
def find_max_prop(arr, proposals):   
    rate_list = []    
    for idx ,proposal in enumerate(proposals):  
        rate_inter = compute_rate(arr , proposal)
        rate_arr = compute_rate(proposal , arr)
        if rate_inter > 0 and (float(np.sum(proposal)) / float(proposal.shape[0] * proposal.shape[1])) < 0.6:   # remove proposal area larger than 0.6 img_size
            if rate_arr > 0.8:            
                rate_list.append([rate_inter , idx])
    
    if rate_list == []:   # reserve seed in this situation
        return arr , int(np.sum(arr))
    else:
        prop_index = rate_list[rate_list.index(max(rate_list))][1]   # get max rate 's proposal index
        return proposals[prop_index] , int(np.sum(proposal[prop_index]))

def takefirst(elem):
    return elem[0]

# use all_cls_heatmaps to remove invalid expanded area caused by sal uninterested area
def all_heat_remove( seg ,up_total_heatmaps ):
    h , w = seg.shape
    for h_idx in range(h):
        for w_idx in range(w):
            all_class_scores = up_total_heatmaps[: , h_idx , w_idx]   # class order 0-19
            pixel_cls = seg[h_idx][w_idx]     
            max_value_cls = np.argmax(all_class_scores) + 1 
            if pixel_cls != 0:
                if pixel_cls != max_value_cls:
                    seg[h_idx][w_idx] = 0
    return seg

def cls_heat_remove( seg, up_class_heatmaps , gt_cls):
    h , w = seg.shape
    for h_idx in range(h):
        for w_idx in range(w):
            cls_scores = up_class_heatmaps[: , h_idx , w_idx]
            pixel_cls = seg[h_idx][w_idx]
            max_value_cls = gt_cls[np.argmax(cls_scores)]
            if pixel_cls !=0:
                if pixel_cls != max_value_cls:
                    seg[h_idx][w_idx] = 0
    return seg

                

# use cls_heatmaps to deal with multi seed intersection with one sal_cc
def heat_multiseed( seg , sc, up_class_heatmaps , gt_cls):
    pos_x , pos_y = np.where(sc == 1 )
    for idx in range(len(pos_x)):
        cls_values = up_class_heatmaps[: , pos_x[idx] , pos_y[idx]]
        cls_index = np.argmax(cls_values)
        cls = gt_cls[cls_index]
        seg[pos_x[idx]][pos_y[idx]] = cls

    return seg
    
def heat_unused_seed(seg, proposal , up_total_heatmaps, unused_seed_gt , gt_cls):
    pos_x , pos_y = np.where(proposal == 1)
    for idx in range(len(pos_x)):
        cls_values = up_total_heatmaps[: , pos_x[idx] , pos_y[idx]]
        cls = np.argmax(cls_values) + 1
        if cls == unused_seed_gt :
            seg[pos_x[idx]][pos_y[idx]] = cls
    return seg


# for seed, one class has several regions, for sal only one class and several regions
# get binary arr , return list of connected component region for each class in seed or saliency
def compute_cc(arr ,  minarea =300):
    labels = measure.label(arr , connectivity = 2)
    
    dst = morphology.remove_small_objects(labels , min_size = minarea ,  connectivity = 2)
    labels = measure.label(dst, connectivity = 2)   
    prop = measure.regionprops(labels)
# get connectivity region list
    region_list = [] 
    region_size_list = []
    for idx, idx_region in enumerate(prop):  # whether including bg for label 0, no
        region = np.zeros((arr.shape))
        one_num = 0
        for pixel_coord in idx_region.coords:
            region[pixel_coord[0]][pixel_coord[1]] = 1
            one_num = one_num +1
        region_list.append(region)
        region_size_list.append(one_num)
    return region_list , region_size_list

# get superpixel small connectivity region area
#    small_region_list = []
#    for region in region_list:
#        has_in = False    # to judge where superpixel has in connected region
#        
#        for cluster in superpixel.clusters:
#            small_region =  np.zeros((arr.shape)) 
#            for pixel in cluster.pixels:
#                x, y = pixel
#                if region[x][y] == 1:
#                    has_in = True
#                    small_region[x][y] = 1
#            if has_in == True:
#                small_region_list.append(small_region)
#                
#    print('this class include small superpixel connected regions ')
#    print(len(small_region_list))
#    return small_region_list

#################################################### util funciton ############################################################################

def compute_rate(arr1, arr2):
    I = float(np.sum(np.logical_and(arr1 != 0 , arr2 !=0)))
    arr2_area = float(np.sum(arr2 != 0))
    
    
    
    rate = I / arr2_area
    
    return rate

# compute two binary arr iou
def compute_iou(arr_1 , arr_2):
    I = float(np.sum(np.logical_and(arr_1, arr_2)))
    U = float(np.sum(np.logical_or(arr_1, arr_2)))

    if U == 0 :
        U = float(1)
            
    iou = I / U

    return iou

# this two functions is to set value when combining sal and seed
def set4sal_cc(seg, c, old_value = 1, new_value = 0):
    area_x , area_y  = np.where( c == old_value )
    num = len(area_x)
    for idx in range(num):
        seg[area_x[idx]][area_y[idx]] = new_value   
        
    return seg
def set_infer_pred( seg, pred, sc):
    area_x, area_y = np.where(sc ==1)
    num = len(area_x)
    for idx in range(num):
        seg[area_x[idx]][area_y[idx]] = pred[area_x[idx]][area_y[idx]]
    return seg

#######################################################  not be tested now ######################################################################################
# only foreground label wiil be updated and expand
def iter_1_seed_superpixel( p , seed , rate = 0.2):
    h , w = seed.shape
    super_seed = np.zeros((h,w),dtype = np.uint8)
    
    for cluster in p.clusters:
        cluster_labels = []
        low_prop_num = rate * len(cluster.pixels)
        label_num ={}    # for >= 0.1 cluster area to propagate
        # first to judge how manny fg labels in this cluster, other than bg
        for pixel in cluster.pixels:
            seed_label = seed[pixel[0]][pixel[1]]
            if seed_label != 0 and seed_label not in cluster_labels:
                cluster_labels.append(seed_label)
                label_num[seed_label] = 1
            if seed_label in cluster_labels:
                label_num[seed_label] = label_num[seed_label] + 1
        
        # for seed and superpixel intersection label is only one
        if cluster_labels != [] :
            if len(cluster_labels) == 1 and label_num[cluster_labels[0]] > low_prop_num :
                for pixel in cluster.pixels:
                    super_seed[pixel[0]][pixel[1]] = cluster_labels[0]
            else:
                for pixel in cluster.pixels:
                    seed_label = seed[pixel[0]][pixel[1]]
                    # for more than 2 labels condition , reserve the initial and randomly propagate to unsure area
                    if seed_label in cluster_labels:
                        super_seed[pixel[0]][pixel[1]] = seed_label
                    else:
                        super_seed[pixel[0]][pixel[1]] = choice(cluster_labels)          
    return super_seed


# initial saliency update algorithm
def update_sal(file_name, img, anno, pred, cam, a = 15, b = 40):
    c_anno = dense_crf(img, anno)
    c_pred = dense_crf(img, pred)
    c_cam = dense_crf(img, cam)
    #discard_list = []
    if MAE(c_anno, c_pred) <= a:
        s_update = dense_crf(img, (c_anno + c_pred)/2)
    elif MAE(c_anno, c_cam) > b and MAE(c_pred, c_cam) > b:
        s_update = c_anno
        #discard_list.append(file_name)
    elif(MAE(c_anno, c_cam) < MAE(c_pred, c_cam)):
        s_update = c_anno
    elif(MAE(c_anno, c_cam) >= MAE(c_pred, c_cam)):
        s_update = c_pred
    return s_update#, discard_list



if __name__ == '__main__':
#    arr1 = np.array([[0,1,2,3,0],[0,0,0,2,3]])
#    arr2 = np.array([[1,0,1,1,0],[0,1,0,0,1]])
#    iu = compute_rate(arr1 , arr2)
#    print(iu)
 
    palette = make_palette(21).reshape(-1)   

    path1 = '/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/fake_train_aug_class/2007_000504.png'
    path2 = '/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/fake_train_aug_class2/2007_000504.png'
    gt_cls = [15,10]
    new = inter_gt(path1 , path2 , gt_cls)
    new = Image.fromarray(new.astype(np.uint8))
    new.putpalette(palette)
    new.save('/home/hhhhhhhhhh/test.png')

