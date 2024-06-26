# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:55:51 2019

@author: Tanzeel
"""

import torch.nn as nn
import torch.nn.functional as F 
import torch
#Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
#https://arxiv.org/abs/1908.08681v1
#implemented for PyTorch 

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))