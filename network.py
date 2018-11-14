#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
reload(sys)  
sys.setdefaultencoding('utf8') 
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torchvision.models as models
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
from config import config


class DropBlock(nn.Module):
    def __init__(self, drop_prob, block_size, feat_size):
        super(DropBlock, self).__init__()
        assert feat_size > block_size, \
            "block_size can't exceed feat_size"
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.feat_size = feat_size
        self.gamma = self._compute_gamma()
        self.bernouli = Bernoulli(self.gamma)

    def forward(self, x, block_size , feat_size , drop_rate = 0.1):
        # shape: (bsize, channels, height, width)    feature
        
        self.block_size = block_size
        self.feat_size = feat_size
        self.set_drop_probability(drop_rate)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training:
            return x
        else:
            mask = self.bernouli.sample((x.shape[-2], x.shape[-1]))
            if torch.sum(mask!=0) == 0:  # for not drop any area case
                return x
            block_mask = self._compute_block_mask(mask)   # feat size
            block_mask = block_mask.cuda(int(str(x.device)[-1]))    # same gpu as x
            out = x * block_mask[None, None, :, :]
            out = out * block_mask.numel() / block_mask.sum()
            return out
            

    def _compute_block_mask(self, mask):
        height, width = mask.shape

        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                torch.arange(self.block_size).repeat(self.block_size)
            ]
        ).t()

        non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1) # ???????????
        offsets = offsets.repeat(1, nr_blocks).view(-1, 2)

        block_idxs = non_zero_idxs + offsets
        padded_mask = F.pad(mask, (0, self.block_size, 0, self.block_size))
        
        padded_mask[block_idxs[:, 0], block_idxs[:, 1]] = 1.
        block_mask = padded_mask[:height, :width]

        return 1 - block_mask

    def _compute_gamma(self):
        return (self.drop_prob / (self.block_size ** 2)) * \
               ((self.feat_size ** 2) / ((self.feat_size - self.block_size + 1) ** 2))

    def set_drop_probability(self, drop_prob):
        self.drop_prob = drop_prob
        self.gamma = self._compute_gamma()
        self.bernouli = Bernoulli(self.gamma)


class GAP_HighRes(nn.Module):
	
    def __init__(self, config , split = 'train',  pretrained = True, **kwargs):
        super(GAP_HighRes, self).__init__()
        self.split = split
        self.config  = config
        self.pretrained = pretrained
        self.vgg16 = models.vgg16(pretrained = self.pretrained, **kwargs)
        self.same1 = nn.Sequential(*(list(self.vgg16.children())[0])[0:23])        
        self.same2 = nn.Sequential(*list(self.vgg16.children())[0][24:30])
        
        self.fc67 =  nn.Sequential(
                nn.Conv2d(512 ,1024 , kernel_size = 3, padding = 1), nn.ReLU(inplace = True),
                nn.Conv2d(1024 ,1024 , kernel_size = 3, padding = 1), nn.ReLU(inplace = True))
        self.feature = nn.Sequential(*(list(self.same1.children())+list(self.same2.children())+list(self.fc67.children())))
        self.GAP = nn.AvgPool2d(40)
        self.fc = nn.Linear(1024, config.num_classes)
        self.drop_block = DropBlock(0.1 , 20 , 40)  
        print(self.feature)
        
    
        if self.pretrained ==  True:
            self.__initialize_weights(self.fc67)
            self.__initialize_weights(self.GAP)
            self.__initialize_weights(self.fc)
        
    def forward(self, input, drop_rate):
        x = self.feature(input)
        if self.split == 'train':
            x = self.drop_block(x, block_size = x.size(3)//2, feat_size = x.size(3), drop_rate = drop_rate)
            
        x = self.GAP(x)   
        x.contiguous()
        x = x.view(x.size(0), -1)  # -1 means squeeze all other dimension, witch is 1024, 1,1
        x = self.fc(x)
        
        return x
        
    def __initialize_weights(self, model):
    	for m in model.modules():
    		if isinstance(m, nn.Conv2d):
    			n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
    			m.weight.data.normal_(0, math.sqrt(2. /n))
    			if m.bias is not None:
    				m.bias.data.zero_()
    		elif isinstance(m, nn.BatchNorm2d):
    			m.weight.data.fill_(1)
    			m.bias.data.zero_()
    		elif isinstance(m, nn.Linear):
    			n = m.weight.size(1)
    			m.weight.data.normal_(0, 0.01)
    			m.bias.data.zero_()

if __name__ == "__main__":
#    config = 0
    highres = GAP_HighRes(config)
    print(highres)






