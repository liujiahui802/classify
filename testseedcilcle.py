#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# original image input , not resize to 320
import sys
sys.path.append('../')
reload(sys)  
sys.setdefaultencoding('utf8')   # about python2.7 encoding problem whtch not in python3
import torch
import os.path as osp
import numpy as np
import cv2
from PIL import Image

from config import config
from seed.network import GAP_HighRes
from preprocess import Seed_ValPre
from get_label import get_label_batch
from lib.utils.pyt_utils import ensure_dir
from metric import evaluate_single_segmentation
from slic import  SLICProcessor
from update import iter_1_seed_superpixel, seed_dense_crf, sal_seed1, sal_seed2 , sal_seed_mcg1, comb_seed , CLS_seed_chose
from util import make_palette , get_new_dict , get_txt_weightbias
from lib.datasets.voc.voc import Seed_VOC
from visualize import Visual_GAP , visualize_pr , visualize_iou
from seed_util import get_conf, intersec_seed_35, _20_1up_heatmaps , _20_1threshold, _cls_1up_heatmaps , squeeze2 , squeeze3 , squeeze4, squeeze5, squeeze6, squeeze7,  read_initiaL_sal , get_num4TFPN , get_metric
from proposal import img_proposal
from deeplab_stuff.stuff_bg import bg_stuff
from Resnet import resnet50



import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3


def error_analyse(hist):
    fg_error_list = [] 
    h , w = hist.shape
    for i in range(h):
        if i != 0:
            fg_error = []
            ei0 = float(hist[i][0]) / float(np.sum(hist[i]))
            eie = float(np.sum(hist[i]) - hist[i][0] - hist[i][i]) / float(np.sum(hist[i]))
            fg_error.append([ei0 , eie])
            fg_error_list.append(fg_error) 
    
    return fg_error_list



def dense_crf(img, output_probs):
    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q

def fast_hist(a, b, n):   # a for gt , b for predicted output.flatten()
    k = (a >= 0) & (a < n)   # pixel numbers for all class in gt, return array with True and False , n = 21 , leave out 255 ,  k is with size img_h * img_width
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n) 

def get_iou(pred,gt):
    if pred.shape!= gt.shape:
        print 'pred shape',pred.shape, 'gt shape', gt.shape
    assert(pred.shape == gt.shape)    
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    count = np.zeros((21,))  # 21 class for iou rate
    for j in range(21):  # for each class in 21 , GT_idx_j may be 0
        x = np.where(pred==j)     # for place axis x, y
        p_idx_j = set(zip(x[0].tolist(),x[1].tolist()))    
        # x[0] is the x axis , y  is the y axis ,  set return unrepeated all x,y tuple axis list
        x = np.where(gt==j)
        GT_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        #pdb.set_trace()     
        n_jj = set.intersection(p_idx_j,GT_idx_j)
        u_jj = set.union(p_idx_j,GT_idx_j)
        # set calculation for intersevtion and union with class pixel axises
          
        # only  calculate for classes has in gt
        if len(GT_idx_j)!=0:  
            count[j] = float(len(n_jj))/float(len(u_jj))
    result_class = count
    Aiou = np.sum(result_class[:])/float(len(np.unique(gt))) 
    
    return Aiou  # average iou for all  classes whitch is in gt



    


if __name__ == '__main__':
    torch.cuda.set_device(1)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    palette = make_palette(20).reshape(-1)   
    data_setting = {'img_root': config.img_root_folder,  'gt_root': config.gt_root_folder, 'train_source': config.train_source, 'eval_source': config.eval_source}
    deeplab_config = '/home/hhhhhhhhhh/desktop/project_seg/deeplab_stuff/config/cocostuff164k.yaml'    
    deeplab_model_path = '/home/hhhhhhhhhh/desktop/project_seg/deeplab_stuff/data/models/deeplab_resnet101/cocostuff164k/cocostuff164k_iter100k.pth'
    modi_label_dict =  {0:0 , 5:1 , 2:2 , 16:3 , 9:4 , 44:5 , 6:6 , 3:7 , 17:8 , 62:9 , 21:10 , 67:11 , 18:12 , 19:13 , 4:14 ,
                        1:15 , 64:16 , 20:17 , 63:18, 7:19 , 72:20 } 
    
    scores_list = []
    class_scores_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    fg_iou_list = []
    bg_iou_list = []
    class_iou_add = np.zeros((2,21)) 
                                 # first row cal the added iou , second row cal the number to divide
#    model = resnet50(pretrained=False, split = 'train', class_num = 20 )

    model = GAP_HighRes(config, split = 'val', pretrained = True)
    state_dict = torch.load(config.load_path, map_location = lambda storage, loc: storage)
    print('============================')
    print(len(state_dict))
    state_dict = get_new_dict(state_dict)
    print('----------------------------')
    print(len(state_dict))
    model.load_state_dict(state_dict)
    model.cuda()
    
    fc_weight, fc_bias = get_txt_weightbias(config.weight_path)

    test_preprocess = Seed_ValPre(config.image_mean, config.image_std, config.target_size)
    dataset = Seed_VOC(data_setting, "val", test_preprocess)
    num_data = len(dataset)
    print(num_data)
    hist = np.zeros((21,21))
    pytorch_list = []
#    pred_real_all_classes = np.zeros((21, 4))  # 1,2,3,4 for TP,TN , FP, FN
#    f = open('/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/Baseline/seed_cls_heatmaps_single_label/single_list.txt','a')
    for idx in range(num_data):
        sample = dataset[idx]
        gt = sample['mask']
        gt_arr = np.array(gt, dtype = np.uint8) 
        batch_file = sample['batch_files']
        idx_label = get_label_batch([batch_file])[0]                              # size is (1,20)-> (20, )  no bg
        gt_cls = (np.where(idx_label == 1)[0] +1 ).tolist()                      # 1-20
        # img
        img = sample['data']                                                       # ( 3, 321, 321)    tensor ,  no cuda
        img_arr = np.array(img, dtype = np.uint8)
        img_feed = img.view(1,img.size(0),img.size(1), img.size(2))                # from 3,320, 320 to 1,3,320, 320
        img_feed = img_feed.cuda()
        
        print(idx)
        print(batch_file)
#        if len(gt_cls)==1 :
        gt4iou = np.array(Image.open(osp.join('/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/SegmentationClass',batch_file+'.png')).convert(mode = 'P'), dtype = np.uint8)

#        print(idx_label)
#        print('feeding img......')
#        if batch_file == '2011_002578':
        
#        proposals = img_proposal(batch_file , top_num = 10)  # 0 / 1  , and visual image is 0 / 255
        output = model.feature(img_feed)

#        output = model.layer4(img_feed)
        output = output.view(output.size(1),output.size(2), output.size(3))   # from 1, 1024,40,40 to 1024,40,40  # not 40 square

        
######################################################################################################################################
######################################################################################################################################
###########################################################################   20-1 ##################################################
# initial seed
#        up_total_heatmaps, confidence = _20_1up_heatmaps(img_feed, output,fc_weight)   
# squeeze 21 to 1 map and assign label
#        squeezed_map =  np.argmax(up_total_heatmaps, axis = 0 )               #  class 0-19
#        squeezed_map = squeezed_map +1                                        # class 1-20 , and 0 for bg
#        squeezed_map = _20_!threshold(config.threshold, squeezed_map, idx_label, confidence)
#        update_seed = squeezed_map

###########################################################################  cls-1 ################################################## 
# newseed1   
        up_class_heatmaps, cls_max_permap , up_total_heatmaps, all_max_permap  = _cls_1up_heatmaps(img_feed, output,fc_weight, idx_label)

#            np.save('/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/Baseline/seed_cls_heatmaps_single_label/{}.npy'.format(batch_file) , up_total_heatmaps)
#            f.write(batch_file + '\n')
#    f.close()
# save for all heatmaps
                
        
   # attention max may be under 0
#        Visual_GAP(output.detach().cpu().numpy() , batch_file , config.visual_path)
#        ensure_dir(config.visual_path)
#        Visual_GAP(up_total_heatmaps , batch_file , config.visual_path)
        #squeezed_map = squeeze2( up_class_heatmaps, cls_max_permap , idx_label, threshold = config.threshold)
#        squeezed_map = squeeze3( up_total_heatmaps, all_max_permap , idx_label)
#        squeezed_map = squeeze4( up_class_heatmaps , all_max_permap , idx_label)
        squeezed_map = squeeze5(up_class_heatmaps , cls_max_permap, idx_label , threshold = config.threshold)  
#        squeezed_map = squeeze6(all_max_permap , cls_max_permap , up_class_heatmaps , gt_cls)
#        squeezed_map = squeeze7(up_class_heatmaps , gt_cls)


       

#high seed3 to intersection with seed5 , 
#        squeezed_map3 = squeeze3( up_total_heatmaps, all_max_permap , idx_label)
#        squeezed_map5 = squeeze5(up_class_heatmaps , cls_max_permap, idx_label , threshold = config.threshold)
#        squeezed_map = intersec_seed_35(squeezed_map3, squeezed_map5)
        
#  high seed5 intersection with seed
#        squeezed_map3 = squeeze5( up_class_heatmaps, cls_max_permap , idx_label, threshold = 0.5)
#        squeezed_map5 = squeeze5(up_class_heatmaps , cls_max_permap, idx_label , threshold = config.threshold)
#        squeezed_map = intersec_seed_35(squeezed_map3, squeezed_map5)
        

        update_seed = squeezed_map.astype(int)
##########################################################################  seed + saliency #########################################
# seed saliency
#        sal = read_initiaL_sal(batch_file)
#        update_seed = sal_seed1(sal, update_seed , img_arr ,up_class_heatmaps ,  up_total_heatmaps , cls_max_permap ,gt_cls )
#        update_seed = sal_seed2(sal, update_seed, up_class_heatmaps , up_total_heatmaps , cls_max_permap , gt_cls)
#        update_seed = sal_seed_mcg1(sal, update_seed , proposals ,img_arr ,up_class_heatmaps , up_total_heatmaps ,gt_cls )

###########################################################################  superpixel ################################################   
#        p = SLICProcessor(osp.join(config.img_root_folder, (batch_file + '.jpg')), 500, 40)
#        p.iterate_10times()
#        update_seed = iter_1_seed_superpixel(p, update_seed)
        #update_seed = sal_seed(sal, update_seed , img_arr , gt_cls ,p)  

#############################################################################  #  get fg_ bg confidence for each image mask pixel ######################

#        conf4fgbg = get_conf(up_total_heatmaps , squeezed_map, gt_cls)


###############################################################   coco stuff as bg 0 , ######################################################################## 
        
        
#        image_path = osp.join(config.img_root_folder, (batch_file + '.jpg'))
#        bg_seg = bg_stuff( deeplab_config , image_path , deeplab_model_path , cuda = True  , crf = True , modi_label = modi_label_dict)
##        print('seed img shape')
##        print(img_arr.shape)        
##        
##        print('update_seed.sahpe')        
##        print(update_seed.shape)
##        print('bg_seg shape')
##        print(bg_seg.shape)        
#        
#        update_seed[bg_seg == 0] =0
#        gt_mask = update_seed
                   
###########################################################################  crf  ######################################################
        
        
        
        
#        print(img_arr.shape)
#        crf_update_seed = dense_crf (img_arr , update_prob)
#        update_seed = seed_dense_crf(img_arr.transpose(1,2,0).copy(order='C') , update_seed)
##        
##        
##        
###################################################################   CLS_Crf mask intersec with seed reserve ####################################################################
#        update_seed = CLS_seed_chose(update_seed , squeezed_map,up_class_heatmaps , gt_cls ,  ignore = False)
#        
        
        
        
        
        
############################################################################  new metric  ###################################################            
        print('evaluating metric....')
        hist += fast_hist(gt4iou.flatten(),update_seed.flatten(),21)
        iou_pytorch = get_iou(update_seed , gt4iou)    # average iou for each image on labels in gt , not all 21 labels      
        pytorch_list.append(iou_pytorch)
 
    
        
######################################################################   original metric ##############################################           
        
        
#        #pred_real_all_classes = get_num4TFPN(pred_real_all_classes, idx_label , gt_arr, squeezed_map)
#        glo_acc, cls_acc , prec, rec, f1, fg_iou, bg_iou , class_iou = evaluate_single_segmentation(update_seed , gt.numpy(), 21)
#        
#        scores_list.append(glo_acc)
#        class_scores_list.append(cls_acc)
#        precision_list.append(prec)
#        recall_list.append(rec)
#        f1_list.append(f1)
#        fg_iou_list.append(fg_iou)
#        bg_iou_list.append(bg_iou)
#        for index , cls_iou in enumerate(class_iou):
#            if cls_iou != 0:
#                class_iou_add[0][index] = class_iou_add[0][index] + cls_iou
#                class_iou_add[1][index] = class_iou_add[1][index] + 1
#       
## save image with palatte and visualize    
########################## image  ##################################################################       
#        print('saving images.....')
#        img_path = config.img_root_folder
#        gt_path = config.gt_root_folder        
#        
#        img = Image.open(osp.join(img_path, batch_file)+'.jpg')
#        img.putalpha(255)
######################### seed ##################################################################
#        seed = Image.fromarray(np.uint8(squeezed_map))
#        seed.putpalette(palette)
#        seed.putalpha(255)
#        img_seed = Image.blend(img , seed , config.alpha)
#        img_seed_arr = np.array(img_seed, dtype = np.uint8)
#        
##############################  update_superseed ##################################################
        update_seed = Image.fromarray(np.uint8(update_seed))
        update_seed.putpalette(palette)


#        update_seed.putalpha(255)
#
#        
#        img_update_seed = Image.blend(img , update_seed , config.alpha)
#        img_update_seed_arr = np.array(img_update_seed, dtype = np.uint8)
######################   bg stuff seg ##########################################################       
##        bg_seg = Image.fromarray(np.uint8(bg_seg))
#
#        
#        
######################   superseed crf###########################################################
#        crf_update_seed = Image.fromarray(np.uint8(crf_update_seed))
#        crf_update_seed.putpalette(palette)
#        crf_update_seed.putalpha(255)
#        img_crf_update_seed = Image.blend(img , crf_update_seed , config.alpha)
#        img_crf_update_seed_arr = np.array(img_crf_update_seed, dtype = np.uint8)
########################   gt   #################################################################
#        gt = Image.fromarray(gt_arr)       # save image with palatte and visualize
#        gt.putpalette(palette)
#        gt.putalpha(255)
#        gt_arr = np.array(gt, dtype = np.uint8)/home/hhhhhhhhhh/test4error_/SE5S1CRF_m1
##
##################################################################################################
## saving 
#        gt_seed_img_arr = np.concatenate([gt_arr, img_crf_update_seed_arr ], 1)
#        gt_seed_img = Image.fromarray(np.uint8(gt_seed_img_arr))
        
# save for training mask        
#        gt_seed_img = Image.fromarray(np.uint8(gt_mask))
#        gt_seed_img.putpalette(palette)
  #      
        
#        print(fg_iou)
        save_path = '/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/Baseline-Dropblock/feature-0.9-20'
##        
###        save_path = '/home/hhhhhhhhhh/desktop/dataset/aug_voc10582/fake_train_aug_class'
        ensure_dir(save_path)       
##        gt_seed_img.save(osp.join(save_path, '{}.png').format(batch_file)) 
##        bg_seg.save(osp.join(save_path, '{}bg.png').format(batch_file))
        update_seed.save(osp.join(save_path , '{}.png'.format(batch_file)))
#        gt_seed_img.save(osp.join(save_path , '{}-crf.png'.format(batch_file)))
#        np.save(osp.join(save_path,'{}.npy'.format(batch_file)) , conf4fgbg)
 

# new metric
    miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('class iou ',iter , "---" , miou)    
    
    print 'evalpyt2-pytorch',iter,"Mean iou = ",np.sum(miou)/len(miou)
    
    print 'evalpyt1-pytorch',iter, np.sum(np.asarray(pytorch_list))/len(pytorch_list)   # average

#    error_list = error_analyse(hist)
#
#    print(error_list)






       
# metric 
#    score = np.mean(scores_list)
#    class_avg_scores = np.mean(class_scores_list, axis=0)
#    precision = np.mean(precision_list)
#    recall = np.mean(recall_list)
#    f1 = np.mean(f1_list)
#    fg_iou = np.mean(fg_iou_list)
#    bg_iou = np.mean(bg_iou_list)
#    class_iou = class_iou_add[0] / class_iou_add[1]
#    print(fg_iou) 
    
#    precision, recall  = get_metric(pred_real_all_classes)
#    visualize_iou( fg_iou , bg_iou , class_iou, class_names = Seed_VOC.get_class_names(), path = save_path)               
