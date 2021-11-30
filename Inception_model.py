# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 19:16:46 2019

@author: Tanzeel
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from mish import Mish

act_fn=Mish()

class Inception(nn.Module):
    def __init__(self,dropout=0.2):
        super(Inception, self).__init__()
        '''
        self.conv9x9 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(16,eps=0.001))
        '''
       
        self.conv9x9=BasicConv1d(1, 16, kernel_size=9,stride=1,padding=4)
        #self.dropout=nn.Dropout(p=0.05)
        
        self.Module_1=InceptionA(16,32)
        self.fc1 = nn.Linear(408*160, 1)
        #self.fc2 = nn.Linear(2048,1024)
        #self.fc3 = nn.Linear(1024,1)
    def forward(self, x):
            # input x : 1x1x513
            # expected conv1d input : minibatch_size x num_channel x length

            x = x.view(x.shape[0], 1,-1)
            # x : 8 x 1 x 513
            out=self.conv9x9(x)
            #out=self.dropout(out)
            
            out = self.Module_1(out)
            out = out.view(x.size(0),-1 )
            out = self.fc1(out)
            #out = self.fc2(out)
            #out = self.fc3(out)
            return out



class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch9x9_1 = BasicConv1d(in_channels, 32, kernel_size=1)
        self.branch9x9_2 = BasicConv1d(32, 32, kernel_size=9, padding=4)

        self.branch5x5_1 = BasicConv1d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = BasicConv1d(16, 32, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv1d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv1d(16, 32, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv1d(32, 64, kernel_size=3, padding=1)
        
        #self.branch3x3b2_1 = BasicConv1d(in_channels, 16, kernel_size=1)
        #self.branch3x3b2_2 = BasicConv1d(16, 32, kernel_size=5, padding=2)

        self.branch_pool = BasicConv1d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch9x9 = self.branch9x9_1(x)
        branch9x9 = self.branch9x9_2(branch9x9)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        
        #branch3x3_b2 = self.branch3x3b2_1(x)
        #branch3x3_b2 = self.branch3x3b2_2(branch3x3_b2)

        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch9x9, branch5x5, branch3x3dbl,  branch_pool]
        return torch.cat(outputs, 1)

class BasicConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return act_fn(x)
